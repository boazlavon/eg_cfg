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
