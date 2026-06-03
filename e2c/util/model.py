from transformers import LogitsProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import numpy as np
import statistics


def compute_confidence_metrics(confidences, chunk_size=100):
    confidences = np.array(confidences)
    n = len(confidences)
    chunk_means = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_means.append(confidences[start:end].mean())
    return {
        "seq_avg_confidence": confidences.mean(),
        "chunk_avg_confidences": chunk_means,
        "last_chunk_confidence": chunk_means[-1] if chunk_means else None,
        "min_chunk_confidence": min(chunk_means) if chunk_means else None,
    }


class ForceNextTokenProcessor(LogitsProcessor):
    def __init__(self, trigger_token_id, forced_token_id):
        self.trigger_token_id = trigger_token_id
        self.forced_token_id = forced_token_id
        self.active = False

    def __call__(self, input_ids, scores):
        last_token_id = input_ids[0, -1].item()
        if last_token_id == self.trigger_token_id:
            self.active = True
        if self.active:
            mask = torch.full_like(scores, float("-inf"))
            mask[..., self.forced_token_id] = 0
            scores = mask
            self.active = False
        return scores


class TemperatureProcessor(LogitsProcessor):
    """Use T1 before <EXECUTION>, T2 after."""

    def __init__(self, tokenizer, stop_token, T1=1.3, T2=1.0):
        self.tokenizer = tokenizer
        self.stop_token_id = tokenizer.convert_tokens_to_ids(stop_token)
        self.T1 = T1
        self.T2 = T2
        self.use_T2 = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.size(0)
        if self.use_T2 is None:
            self.use_T2 = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        last_token_ids = input_ids[:, -1]
        self.use_T2 |= (last_token_ids == self.stop_token_id)
        T = torch.where(self.use_T2, torch.tensor(self.T2, device=scores.device),
                        torch.tensor(self.T1, device=scores.device))
        return scores / T[:, None]


def merge_model(model_path, world_size=4):
    ckpts = {}
    test_file = os.path.join(model_path, f'model_world_size_{world_size}_rank_0.pt')
    if not os.path.exists(test_file):
        world_size = 8 if world_size == 4 else 4

    shard_files = [os.path.join(model_path, f'model_world_size_{world_size}_rank_{i}.pt') for i in range(world_size)]
    for file_path in shard_files:
        if not os.path.exists(file_path):
            break
        tensors = torch.load(file_path, weights_only=False)
        for n, p in tensors.items():
            p = p.to_local()
            p = torch.tensor(p)
            if n not in ckpts:
                ckpts[n] = p
            else:
                ckpts[n] = torch.cat([ckpts[n], p], dim=0)
    torch.save(ckpts, os.path.join(model_path, 'full.safetensors'))
    return ckpts


def check_resume(path, seed, rank):
    result_file = os.path.join(path, f"result_{seed}_rank{rank}.json")
    static_file = os.path.join(path, f"static_{seed}_rank{rank}.json")
    if os.path.exists(result_file) and os.path.exists(static_file):
        with open(result_file, "r") as f:
            results = json.load(f)
        with open(static_file, "r") as f:
            static = json.load(f)
        print("[Rank {}] Resuming from existing results: {} samples found.".format(rank, len(results)))
        return results, static
    return [], {}


import re


def get_max_step(text: str) -> int:
    matches = re.findall(r'\n\b(\d+)[\.\)]', text)
    numbers = [int(num) for num in matches]
    return max(numbers) if numbers else 0


class EntropyCalculator:
    """Compute entropy statistics over token segments delimited by special tokens."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_token_id(self, token_str):
        if token_str in ('begin', 'end'):
            return None
        return self.tokenizer.convert_tokens_to_ids(token_str)

    def calculate_entropy_stats(self, sample, start_token, end_token):
        tokens, entropies, confidences = sample

        start_id = self.get_token_id(start_token)
        if start_token == 'begin':
            start_idx = 0
        elif start_id in tokens:
            start_idx = tokens.index(start_id)
        else:
            return None

        if end_token == 'end':
            end_id = self.tokenizer.eos_token_id
        else:
            end_id = self.get_token_id(end_token)

        if end_id in tokens:
            end_idx = tokens.index(end_id)
        else:
            return None

        if self.tokenizer.eos_token_id in tokens:
            valid_end_pos = tokens.index(self.tokenizer.eos_token_id)
        else:
            valid_end_pos = len(tokens) - 1

        if end_idx <= start_idx:
            return None

        segment_entropies = entropies[start_idx:end_idx + 1]
        length = len(segment_entropies)
        total_entropy = sum(segment_entropies)
        avg_entropy = total_entropy / length if length > 0 else 0
        max_step = get_max_step(self.tokenizer.decode(tokens[start_idx:end_idx + 1]))
        return {
            "length": length,
            "total_entropy": total_entropy,
            "avg_entropy": avg_entropy,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "max_step": max_step,
            "confidences": compute_confidence_metrics(confidences[start_idx:end_idx + 1]),
            "entropies": entropies[0:valid_end_pos + 1],
            "confidences_all": confidences[0:valid_end_pos + 1],
        }

    def calculate_batch_entropy_stats(self, sample, start_token, end_token):
        all_tokens, all_entropies, all_confidences = sample
        stats_list = []
        for tokens, entropies, confidences in zip(all_tokens, all_entropies, all_confidences):
            stats = self.calculate_entropy_stats((tokens, entropies, confidences), start_token, end_token)
            if stats is not None:
                stats_list.append(stats)
        if not stats_list:
            return {"avg_length": 0, "avg_total_entropy": 0, "avg_entropy_per_token": 0, "sample_count": 0}
        return {
            "avg_length": statistics.mean([s["length"] for s in stats_list]),
            "avg_total_entropy": statistics.mean([s["total_entropy"] for s in stats_list]),
            "avg_entropy_per_token": statistics.mean([s["avg_entropy"] for s in stats_list]),
            "sample_count": len(stats_list),
        }


def load_model(cfg, device="cuda"):
    subfolder = getattr(cfg, "subfolder", None) or None
    extra = {"subfolder": subfolder} if subfolder else {}
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        **extra,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True, **extra)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.type == "fsdp":
        checkpoints = merge_model(cfg.checkpoint_path)
        print("Loading checkpoint from:", cfg.checkpoint_path)
        model.load_state_dict(checkpoints, strict=False)
    elif cfg.type == "lora":
        raise NotImplementedError
    return model, tokenizer
