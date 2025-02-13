### Contains functions for plotting head detection scores and attention patterns ###

import torch
import einops
import pysvelte

def plot_head_detection_scores(
    scores: torch.Tensor,
    zmin: float = -1,
    zmax: float = 1,
    xaxis: str = "Head",
    yaxis: str = "Layer",
    title: str = "Head Matches"
) -> None:
    from neel_plotly import imshow
    imshow(scores, zmin=zmin, zmax=zmax, xaxis=xaxis, yaxis=yaxis, title=title)

def plot_attn_pattern_from_cache(model, cache, prompt, layer_i: int):
    attention_pattern = cache["pattern", layer_i, "attn"].squeeze(0)
    attention_pattern = einops.rearrange(attention_pattern, "heads seq1 seq2 -> seq1 seq2 heads")
    print(f"Layer {layer_i} Attention Heads:")
    return pysvelte.AttentionMulti(tokens=model.to_str_tokens(prompt), attention=attention_pattern)