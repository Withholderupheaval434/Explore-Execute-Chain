"""Model loading for Qwen3 E2C."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device: str = "cuda", torch_dtype=None, subfolder: str = None):
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    attn = "sdpa"
    if torch.cuda.is_available():
        try:
            import flash_attn
            attn = "flash_attention_2"
        except ImportError:
            pass
    extra = {"subfolder": subfolder} if subfolder else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation=attn,
        trust_remote_code=True,
        **extra,
    )
    # Qwen3 ignores top_k; clear it to avoid "generation flags are not valid" warning
    if getattr(model, "generation_config", None) is not None and hasattr(model.generation_config, "top_k"):
        model.generation_config.top_k = None
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, **extra)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
