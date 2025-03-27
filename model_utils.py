import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    return device


def load_model(model_name: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer


def extract_new_tokens(tokenizer, input_ids: torch.Tensor, prompt_input_ids_len) -> str:
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("Expected input_ids to have shape (1, sequence_length)")

    new_token_ids = input_ids[:, prompt_input_ids_len:]
    new_text = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]
    return new_text, new_token_ids


def calculate_tokens_length(tokenizer, prompt):
    prompt_token_ids = tokenizer(prompt, return_tensors="pt")
    prompt_input_ids = prompt_token_ids["input_ids"]  # shape: (1, prompt_len)
    prompt_input_ids_len = prompt_input_ids.shape[1]
    return prompt_input_ids_len
