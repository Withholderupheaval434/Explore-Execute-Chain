import os
import sys
import json
from typing import List
import tqdm
from transformers import AutoTokenizer
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import destroy_process_group
from e2c.util.dataset import save_as_dataset, load_dataset_from_exploration
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
e2c_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, e2c_dir)

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    print("Warning: hydra-core not installed. Using basic config loading.")
    hydra = None

from e2c.util.dataset import max_token_dataset, max_batch_size, is_multi_choice, load_dataset_by_name
from e2c.util.model import load_model

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: VLLM not available. Install with: pip install vllm")


def ddp_setup():
    from torch.distributed import init_process_group
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def decode_with_selected_special_tokens(tokenizer, special_tokens, token_ids):
    """Decode token_ids, keeping only the given special tokens and stripping others."""
    special_tokens += ['<EXPLORATION>', '</EXPLORATION>', '<EXECUTION>', '</EXECUTION>']
    keep_special_token_ids = {tokenizer.convert_tokens_to_ids(token) for token in special_tokens}
    filtered_token_ids = [
        token_id for token_id in token_ids
        if token_id not in tokenizer.all_special_ids or token_id in keep_special_token_ids
    ]
    return tokenizer.decode(filtered_token_ids, skip_special_tokens=False)


def generate_batch_vllm(
    llm_model,
    tokenizer,
    input_texts,
    max_new_tokens=512,
    device="cuda",
    stop_token=None,
    sample_num=1,
    temperature=1.0,
    top_p=1.0,
    enable_thinking=False,
    solution_prefix="",
):
    if not VLLM_AVAILABLE:
        raise ImportError("VLLM not available. Please install with: pip install vllm")

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=sample_num,
        stop=["</s>"] + ([stop_token] if stop_token else []),
    )
    outputs = llm_model.generate(input_texts, sampling_params)
    results = []
    for output in outputs:
        sample_results = [choice.text for choice in output.outputs]
        results.append(sample_results)
    return results


def generate_batch(
    model,
    tokenizer,
    input_text,
    max_new_tokens,
    device="cuda",
    stop_token=None,
    sample_num=1,
    temperature=1.0,
    top_p=0.7,
):
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(device)
    B = len(input_text)
    model.eval()
    model.to(device)

    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_length=input_ids.input_ids.shape[1] + max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=sample_num,
            return_dict_in_generate=True,
            output_scores=False,
            eos_token_id=[tokenizer.eos_token_id] if stop_token is None else [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids(stop_token)[0],
            ],
        )
        sequences = outputs.sequences[:, input_ids.input_ids.shape[1]:]
        B = input_ids.input_ids.shape[0]
        total_size = B * sample_num
        assert sequences.shape[0] == total_size

        all_results = []
        for s in range(sample_num):
            batch_tokens = sequences[s::sample_num]
            batch_texts = []
            for i in range(B):
                token_ids = batch_tokens[i].cpu().tolist()
                text = decode_with_selected_special_tokens(tokenizer, [], token_ids)
                batch_texts.append(text)
            all_results.append(batch_texts)

    return all_results


def get_distributed_dataloader(dataset, batch_size, rank, world_size):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    indices = list(sampler)
    return [dataset[i] for i in indices], len(indices)


def merge_generations_from_all_ranks(save_path, seed, world_size):
    if torch.distributed.get_rank() != 0:
        return

    all_generations = []
    for rank in range(world_size):
        gen_file = os.path.join(save_path, f"generations_{seed}_rank{rank}.json")
        if os.path.exists(gen_file):
            with open(gen_file, "r") as f:
                all_generations.extend(json.load(f))

    if all_generations:
        with open(os.path.join(save_path, f"generations_{seed}_merged.json"), "w") as f:
            json.dump(all_generations, f, ensure_ascii=False, indent=4)
        print(f"Merged {len(all_generations)} samples.")


def check_resume(save_path, seed, rank):
    gen_file = os.path.join(save_path, f"generations_{seed}_rank{rank}.json")
    if os.path.exists(gen_file):
        with open(gen_file, "r") as f:
            return json.load(f)
    return []


@hydra.main(version_base=None, config_path="../../e2c/config", config_name="generate")
def generate_main(cfg):
    if cfg.model.type == "vllm":
        device = "cuda"
        rank = 0
        world_size = 1
        torch.manual_seed(cfg.generation.seed)

        if not VLLM_AVAILABLE:
            raise ImportError("VLLM not available. Please install with: pip install vllm")

        print(f"Loading VLLM model from: {cfg.model.model_path}")
        vllm_config = cfg.model.get('vllm', {})

        num_gpus = torch.cuda.device_count()
        configured_tp_size = vllm_config.get('tensor_parallel_size', -1)
        tensor_parallel_size = num_gpus if configured_tp_size == -1 else min(configured_tp_size, num_gpus)

        print(f"Using {num_gpus} GPUs, tensor_parallel_size={tensor_parallel_size}")

        vllm_kwargs = {
            'model': cfg.model.model_path,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': vllm_config.get('gpu_memory_utilization', 0.85),
            'max_model_len': vllm_config.get('max_model_len', 8192),
            'dtype': vllm_config.get('dtype', 'bfloat16'),
            'trust_remote_code': vllm_config.get('trust_remote_code', True),
        }
        for key in ('max_num_seqs', 'max_num_batched_tokens', 'enforce_eager',
                    'enable_chunked_prefill', 'enable_prefix_caching', 'disable_log_stats'):
            if key in vllm_config:
                vllm_kwargs[key] = vllm_config[key]

        model = LLM(**vllm_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path, trust_remote_code=True)
        print(f"Model loaded ({tensor_parallel_size} GPUs).")

    else:
        ddp_setup()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device = f"cuda:{rank}"
        torch.manual_seed(cfg.generation.seed + rank)

        model, tokenizer = load_model(cfg.model, device)
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank])

    if 'all' in cfg.generation.dataset:
        cfg.generation.dataset = ["gsm8k", "math", "aime24", "aime25", "amc23", "math500", "minerva", "olympiad_bench"]
    if 'med' in cfg.generation.dataset:
        cfg.generation.dataset = ["clinical_knowledge", "college_biology", "college_medicine",
                                   "medical_genetics", "professional_medicine", "anatomy", "medqa", "medmcqa"]
    if 'resume' in cfg.generation.dataset:
        cfg.generation.dataset = [
            os.path.join(cfg.generation.resume_dir, dn, f"generations_{cfg.generation.seed}_merged.json")
            for dn in ["gsm8k", "math", "aime24", "aime25", "amc23", "math500", "minerva", "olympiad_bench"]
        ]
        cfg.generation.dataset = [dn for dn in cfg.generation.dataset if os.path.exists(dn)]

    for dataset_name in cfg.generation.dataset:
        print(f"Loading dataset: {dataset_name}")

        if dataset_name.endswith(".json"):
            dataset, dataset_name = load_dataset_from_exploration(dataset_name)
        else:
            try:
                dataset, dataset_name = load_dataset_by_name(dataset_name)
                print(f"Dataset loaded: {len(dataset)} samples")
            except Exception as e:
                print(f"Failed to load dataset {dataset_name}: {e}")
                continue

        if cfg.generation.get('use_default', False):
            if dataset_name in is_multi_choice:
                cfg.generation.system_prompt = r"You are a medical expert. You will be given a medical question and several candidate answers. Please choose the best answer based on your medical knowledge. You must give your answer in the format: 'The correct answer is boxed{A,B,C or D}'."
                cfg.generation.question_suffix = r"\nPlease reasoning step-by-step.Provide the final answer in the boxed{}."
            else:
                cfg.generation.system_prompt = r""
                cfg.generation.question_suffix = r"\nPlease reasoning step-by-step.Provide the final answer in the boxed{}."

        if cfg.generation.batch_size == -1:
            batch_size = max_batch_size.get(dataset_name, 10) // cfg.generation.sample_num
        else:
            batch_size = cfg.generation.batch_size // cfg.generation.sample_num

        if cfg.generation.max_new_tokens == -1:
            max_tokens = max_token_dataset.get(dataset_name, 1000)
        else:
            max_tokens = cfg.generation.max_new_tokens

        save_path = os.path.join(cfg.generation.save_path, dataset_name)
        os.makedirs(save_path, exist_ok=True)

        if cfg.generation.get('resume', False):
            generations = check_resume(save_path, cfg.generation.seed, rank)
            print(f"[Rank {rank}] Resuming from {len(generations)} existing samples.")
        else:
            generations = []

        if cfg.model.type == "vllm":
            subset = dataset
            subset_size = len(dataset)
            bar = tqdm.tqdm(total=len(dataset), desc=f"Generating {dataset_name}", position=0, leave=True)
            start_idx = len(generations)
        else:
            subset, subset_size = get_distributed_dataloader(dataset, batch_size, rank, world_size)
            bar = tqdm.tqdm(total=len(dataset), desc=f"Generating {dataset_name}", position=0, leave=True) if rank == 0 else None

            local_processed = len(generations)
            init_buf = torch.tensor([local_processed], device=device, dtype=torch.int64)
            torch.distributed.all_reduce(init_buf, op=torch.distributed.ReduceOp.SUM)
            if rank == 0 and init_buf[0].item() > 0:
                bar.n = init_buf[0].item()
                bar.refresh()

            start_idx = len(generations)

        for i in range(start_idx, subset_size, batch_size):
            batch_data = subset[i:i + batch_size]

            system_prompt = []
            enable_thinking = cfg.generation.get("enable_thinking", False)
            if cfg.generation.get("system_prompt", None) is not None:
                system_prompt = [{"role": "system", "content": cfg.generation.system_prompt}]

            questions = [q["question"] + cfg.generation.get('question_suffix', '') for q in batch_data]
            questions = [
                tokenizer.apply_chat_template(
                    system_prompt + [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=cfg.generation.get("enable_thinking", True),
                )
                for q in questions
            ]
            questions = [q + cfg.generation.get("solution_prefix", "") for q in questions]

            if 'prompt' in batch_data[0]:
                questions = [q['prompt'] for q in batch_data]

            if cfg.model.type == "vllm":
                samples = generate_batch_vllm(
                    model, tokenizer, questions,
                    max_new_tokens=max_tokens,
                    device=device,
                    stop_token=cfg.generation.get("stop_token", None),
                    sample_num=cfg.generation.sample_num,
                    temperature=cfg.generation.temperature,
                    top_p=cfg.generation.top_p,
                    solution_prefix=cfg.generation.get("solution_prefix", ''),
                )
            else:
                samples = generate_batch(
                    model.module, tokenizer, questions, max_tokens,
                    device=device,
                    stop_token=cfg.generation.get("stop_token", None),
                    sample_num=cfg.generation.sample_num,
                    temperature=cfg.generation.temperature,
                    top_p=cfg.generation.top_p,
                )

            for batch_idx in range(len(batch_data)):
                if cfg.model.type == "vllm":
                    responses = samples[batch_idx] if batch_idx < len(samples) else []
                else:
                    responses = [samples[s][batch_idx] for s in range(cfg.generation.sample_num)]

                generations.append({
                    "question": batch_data[batch_idx]["question"],
                    "answer": batch_data[batch_idx].get("answer", ""),
                    "prompt": questions[batch_idx],
                    "responses": responses,
                })

            if cfg.model.type == "vllm":
                output_file = os.path.join(save_path, f"generations_{cfg.generation.seed}_merged.json")
                with open(output_file, "w") as f:
                    json.dump(generations, f, ensure_ascii=False, indent=4)
                if bar is not None:
                    bar.n = len(generations)
                    bar.refresh()
            else:
                with open(os.path.join(save_path, f"generations_{cfg.generation.seed}_rank{rank}.json"), "w") as f:
                    json.dump(generations, f, ensure_ascii=False, indent=4)

                local_processed = len(generations)
                buf = torch.tensor([local_processed], device=device, dtype=torch.int64)
                torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
                if rank == 0 and bar is not None:
                    bar.n = buf[0].item()
                    bar.refresh()

        if bar is not None:
            bar.close()

        if cfg.model.type == "vllm":
            final_output_file = os.path.join(save_path, f"generations_{cfg.generation.seed}_merged.json")
            print(f"Generation complete. {len(generations)} samples saved to {final_output_file}")
            if cfg.generation.save_as_dataset:
                save_as_dataset(generations, os.path.join(save_path, f"exploration_{cfg.generation.seed}.parquet"))
        else:
            torch.distributed.barrier()
            merge_generations_from_all_ranks(save_path, cfg.generation.seed, world_size)

    if cfg.model.type != "vllm":
        destroy_process_group()


if __name__ == "__main__":
    generate_main()
