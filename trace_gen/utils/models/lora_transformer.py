from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM


def get_lora_config(
    rank: int = 16, alpha: int = 16, dropout: float = 0.1
):
    return LoraConfig(
        lora_alpha=alpha,
        lora_dropout=dropout,
        r=rank,
        bias="none",
        task_type="CAUSAL_LM",
    )

def add_lora_adapter(
    model: AutoModelForCausalLM, rank: int = 16, alpha: int = 16, dropout: float = 0.1
):
    lora_config = get_lora_config(rank, alpha, dropout)
    model.add_adapter(lora_config)

def merge_lora_adapter(base_model: AutoModelForCausalLM, adapter_path: str, save_path: str = ""):
    model = PeftModel.from_pretrained(base_model, adapter_path, use_safetensors=True)
    merged_model = model.merge_and_unload()
    if save_path:
        merged_model.save_pretrained(save_path)
    return merged_model
