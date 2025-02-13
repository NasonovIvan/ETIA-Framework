### Utility functions for matrix checks and token subsequence search ###

import torch

def is_square(x: torch.Tensor) -> bool:
    """Checks if `x` is a square matrix."""
    return x.ndim == 2 and x.shape[0] == x.shape[1]

def is_lower_triangular(x: torch.Tensor) -> bool:
    """Checks if `x` is a lower triangular matrix."""
    if not is_square(x):
        return False
    return x.equal(x.tril())

def find_subsequence(sequence, subsequence):
    """
    It searches for a sublist of a subsequence in the sequence list.
    Returns the index of the beginning of the first occurrence, or -1 if not found.
    """
    n = len(subsequence)
    for i in range(len(sequence) - n + 1):
        if sequence[i:i+n] == subsequence:
            return i
    return -1

def get_emotion_token_indices(model, prompt: str, emotion: str):
    """
    In a substitution prompt, {emotion} determines the position of the tokens,
    corresponding to the inserted emotion. Returns a list of indexes.
    """
    # We get tokens for full prompt:
    tokens = model.to_tokens(prompt).tolist()

    # We get tokens specifically for the emotion string (without the BOS token):
    emotion_tokens = model.to_tokens(emotion, prepend_bos=False).tolist()

    # We are trying to find tokens based on possible tokenization options.
    variants = [
        emotion_tokens,  # Original tokens
        [emotion],       # Full word
        [emotion.lower()],  # In lowercase
    ]

    for variant in variants:
        start_idx = find_subsequence(tokens, variant)
        if start_idx != -1:
            return list(range(start_idx, start_idx + len(variant)))

    # If the standard options didn't work, we'll try a more flexible search
    emotion_lower = emotion.lower()
    for i in range(len(tokens)):
        token_str = model.to_string(tokens[i]).lower().strip()
        if emotion_lower in token_str:
            return [i]

    raise ValueError(f"Couldn't find the token sequence for the emotion: {emotion}")
