import torch
import json
import hashlib


def format_token(tokenizer, idx):
    return tokenizer.decode(idx).strip() or repr(tokenizer.decode(idx))


def stable_hash(obj):
    s = json.dumps(obj)
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def print_top_k_token_probs(tokenizer, p1, p2, k=5):
    # Get top-k from p1 and p2
    top1_vals, top1_idxs = torch.topk(p1, k)
    top2_vals, top2_idxs = torch.topk(p2, k)

    print(f"\nTop-{k} tokens in orignal_p and guided_p :")
    print(
        f"{'Token (orignal_p)':<15} {'orignal_p':>10}    {'Token (guided_p)':<15} {'guided_p':>10}"
    )
    print("-" * 56)

    for (idx1, val1), (idx2, val2) in zip(
        zip(top1_idxs[0], top1_vals[0]), zip(top2_idxs[0], top2_vals[0])
    ):
        token1 = format_token(tokenizer, idx1)
        token2 = format_token(tokenizer, idx2)
        print(f"{token1:<15} {val1.item():10.4f}    {token2:<15} {val2.item():10.4f}")


# Stay on topic with Classifier-Free Guidance
# https://arxiv.org/abs/2306.17806
# Inputs are the prior probability P and conditional probabilities P_c
# associated with a dynamic signal (condition) c
def apply_guidance(P, P_c, gamma, eps=1e-8, tokenizer=None, debug=False):
    R = torch.ones_like(P)
    R *= (P_c + eps) / (P + eps)

    P_guided = P * R**gamma
    P_guided = P_guided / P_guided.sum(dim=-1, keepdim=True)  # normalize across vocab
    if debug:
        print(f"gamma={gamma}")
        print_top_k_token_probs(tokenizer, P, P_guided, k=3)
    return P_guided


LOG_STABLE_EPS = 1e-10  # ln(1e-10) ~ -23


def mask_topk_probs_log_stable(
    probs: torch.Tensor, k: int, epsilon: float = LOG_STABLE_EPS
):
    topk_vals, topk_indices = torch.topk(probs, k, dim=-1)
    masked_probs = torch.full_like(probs, epsilon)
    masked_probs.scatter_(dim=-1, index=topk_indices, src=topk_vals)
    return masked_probs
