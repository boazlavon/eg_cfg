import torch
import torch.nn.functional as F
from scipy.stats import entropy
from torch.nn.functional import cosine_similarity


def get_probabilities(model, tokenizer, prompt: str, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
    return F.softmax(logits, dim=-1)


def analyze_probability_changes(p1, p2, tokenizer, top_n=20, k=3):
    top_p1_vals, top_p1_idx = torch.topk(p1, top_n)
    top_p2_vals, top_p2_idx = torch.topk(p2, top_n)

    set_p1, set_p2 = set(top_p1_idx[0].tolist()), set(top_p2_idx[0].tolist())
    intersection = list(set_p1 & set_p2)

    increased, decreased = [], []
    for idx in intersection:
        token_str = tokenizer.decode(idx).strip()
        prob_p1, prob_p2 = p1[0, idx].item(), p2[0, idx].item()
        ratio = prob_p2 / (prob_p1 + 1e-8)
        if prob_p2 > prob_p1:
            increased.append((token_str, ratio, prob_p1, prob_p2))
        else:
            decreased.append((token_str, ratio, prob_p1, prob_p2))

    return {
        "top_k_increased": sorted(increased, key=lambda x: x[1], reverse=True)[:k],
        "top_k_decreased": sorted(decreased, key=lambda x: x[1])[:k],
        "top_k_prompt1": [
            (tokenizer.decode(idx).strip(), val.item())
            for idx, val in zip(top_p1_idx[0], top_p1_vals[0])
        ][:k],
        "top_k_prompt2": [
            (tokenizer.decode(idx).strip(), val.item())
            for idx, val in zip(top_p2_idx[0], top_p2_vals[0])
        ][:k],
    }


def distribution_similarity(p1, p2):
    kl_div = entropy(p1.cpu().numpy().flatten(), p2.cpu().numpy().flatten())
    cos_sim = cosine_similarity(p1, p2).item()
    return kl_div, cos_sim


def print_results(prompt1, prompt2, results, kl_div, cos_sim):
    print(f"\nPrompt1: '{prompt1}'")
    print("Top-5 tokens:")
    for token, prob in results["top_k_prompt1"]:
        print(f"  '{token}': {prob:.5f}")

    print(f"\nPrompt2: '{prompt2}'")
    print("Top-5 tokens:")
    for token, prob in results["top_k_prompt2"]:
        print(f"  '{token}': {prob:.5f}")

    print("\nTokens among top-20 in BOTH prompts with INCREASED probability:")
    for token, ratio, p1_val, p2_val in results["top_k_increased"]:
        print(f"  '{token}': ratio={ratio:.2f}, p1={p1_val:.5f}, p2={p2_val:.5f}")

    print("\nTokens among top-20 in BOTH prompts with DECREASED probability:")
    for token, ratio, p1_val, p2_val in results["top_k_decreased"]:
        print(f"  '{token}': ratio={ratio:.2f}, p1={p1_val:.5f}, p2={p2_val:.5f}")

    print("\n📊 Similarity between distributions:")
    print(f"  KL Divergence (p1 || p2): {kl_div:.5f}")
    print(f"  Cosine Similarity: {cos_sim:.5f}")
