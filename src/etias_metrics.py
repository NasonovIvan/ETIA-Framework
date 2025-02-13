### Contains ETIAS metric functions related to emotion tokens and attention scores ###

import numpy as np
from src.utils import get_emotion_token_indices

def compute_etias(model, prompt: str, emotion: str, layer_choice: int = -1):
    """
    Calculates the share of attention (ETIAS) given to tokens,
    appropriate emotions. By default, the last layer of the model is used.

    Args:
        model: LLM model.
        prompt (str): Prompt with emotion substitution.
        emotion (str): Emotion to analyze.
        layer_choice (int): The model layer (-1 means the last layer).

    Returns:
        float: The average share of attention on emotional tokens across all heads of the layer.

    Raises:
        ValueError: If emotional token indexes cannot be found.

    Example Usage:
        etias_score = compute_etias(model, prompt="...", emotion="anger")

        print(f"ETIAS Score for 'anger': {etias_score:.4f}")

    Notes:
      • The self-attention matrix from the selected model layer is used.
      • Attention is normalized throughout the sequence.
      • Only the heads of the selected layer (or all layers) are counted.
    """
    tokens = model.to_tokens(prompt).to(model.cfg.device)

    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    # Extracting the attention matrices for the selected layer
    attn_matrices = cache["pattern", layer_choice, "attn"]  # [n_heads x seq_len x seq_len]

    emo_indices = get_emotion_token_indices(model, prompt, emotion)

    # Compute attention to the emotional tokens for each head.
    scores = []

    for head in range(attn_matrices.shape[0]):
        head_attn = attn_matrices[head]  # [seq_len x seq_len]

        # The summ of attention paid to emotional tokens
        emo_attn = head_attn[:, emo_indices].sum()

        # The total summ of attention over the entire sequence
        total_attn = head_attn.sum()

        scores.append(emo_attn.item() / total_attn.item())

    return np.mean(scores)


def compute_diff_etias(model, prompt_template: str, emotion: str, layer_choice: int = -1):
    """
    Compares the ETIAS for the prompt with the given emotion and the base case ("none").

    Args:
        model: LLM model.
        prompt_template (str): A prompt template with {emotion}.
        emotion (str): Emotion to analyze.
        layer_choice (int): The model layer (-1 means the last layer).

    Returns:
        float: The difference between the ETIAS for a given emotion and "none".

    Example Usage:
        diff_etias_score = compute_diff_etias(model, template="...", emotion="anger")

        print(f"Diff ETIAS Score for 'anger': {diff_etias_score:.4f}")

     Notes:
       • Uses `compute_etias' to calculate ETIAS values.
       • The greater the difference between the values, the stronger the influence of emotion on the model's attention.
     """

    prompt_emotion = prompt_template.format(emotion=emotion)
    prompt_none = prompt_template.format(emotion="none")

    etias_emotion = compute_etias(model, prompt_emotion, emotion, layer_choice=layer_choice)
    etias_none = compute_etias(model, prompt_none, "none", layer_choice=layer_choice)

    return etias_emotion - etias_none

def compute_diff_detection( # THIS METRIC DOES NOT USE IN THE EMOTIONAL PROMPTING ANALYSIS
    model,
    prompt_template: str,
    emotion: str,
    detection_pattern: str = "previous_token_head"
):
    """
    Compares the detection scores for the prompt with a specific emotion and the base prompt (with "none").

    Args:
        model: LLM model.
        prompt_template (str): A prompt template with {emotion}.
        emotion (str): Emotion to analyze.
        detection_pattern (str): The type of detection pattern (for example, "previous_token_head").

    Returns:
        float: The difference between the detection scores for a given emotion and "none".

    Example Usage:
        diff_detection_score = compute_diff_detection(model, template="...", emotion="anger")

        print(f"Diff Detection Score for 'anger': {diff_detection_score:.4f}")

     Notes:
       • Uses the detect_head function to calculate detection scores.
       • The greater the difference between the values, the stronger the influence of emotion on the attention patterns of the model.
     """

    from transformer_lens.head_detector import detect_head

    prompt_emotion = prompt_template.format(emotion=emotion)
    prompt_none = prompt_template.format(emotion="none")

    # We calculate the detection scores for both cases
    head_scores_emotion = detect_head(model, prompt_emotion, detection_pattern)
    head_scores_none = detect_head(model, prompt_none, detection_pattern)

    score_emotion = head_scores_emotion.mean().item()
    score_none = head_scores_none.mean().item()

    return score_emotion - score_none