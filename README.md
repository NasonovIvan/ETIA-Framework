# ETIA Framework

![etias](images/etias_pipeline.pdf)

The Emotion-Aware Textual Influence Activation (ETIA) Framework automatically identifies effective emotional prompts for Large Language Models (LLMs) by quantifying how much they focus on emotion-related tokens. This approach bridges the gap between traditional prompt engineering and the intrinsic mechanisms of model internals, enabling systematic selection of emotional prompts to enhance both model performance and benchmark quality.

## Getting Started

To get started, install the dependencies with Poetry and run the main entry point:

```bash
poetry install
python main.py
```

Or you can try our framework pipeline [notebook in Google Colab](https://colab.research.google.com/drive/1MOITNl9sciA1Pbya-jhSq9rqcAkkB6Df?usp=sharing).

*Do not forget about `config.py` - there are the samples of prompts and emotions. You can use ETIA Framwork for scoring your prompts without emtions categories, but in such case **you have to rewrite the code** of prompts-emotions iteration functions:*

- `src/utils.py/get_emotion_token_indices`
- `src/etias_metrics.py/compute_etias`
- `src/etias_metrics.py/compute_diff_etias`
- `src/evaluation.py/evaluate_emotional_prompts`

## Project Structure

- `main.py` The main entry point for running experiments from the console.

- `config.py` Contains definitions for **PROMPTS** and **EMOTIONS**, making it easy to modify the experimental settings.

- `plotting/plotting.py` Functions to plot head detection scores and visualize attention patterns.

- `src/utils.py` Helper functions for tasks such as token subsequence search and matrix validations.

- `src/head_detection.py` Implements attention head detection methods and different detection patterns.

- `src/etias_metrics.py` Computes key metrics like the ETIA Score, Differential ETIA Score, and other detection metrics.

- `src/evaluation.py` Evaluates the model on each prompt/emotion combination and exports results to CSV files.

- `notebooks/ETIAS_notebook.ipynb` Jupyter Notebook of framework pipeline.

## Overview

Current benchmarks in emotional prompt engineering lack a systematic method to select prompts. The ETIA Framework addresses this by:

- **Analyzing Attention Patterns**:
Using internal activation data (accessed via libraries such as TransformerLens) to measure how much attention is allocated to emotion tokens.

- **ETIA Score Calculation**:
Computing a score based on the model's aggregated normalized attention toward emotion-related tokens across selected layers and heads.

- **Differential Evaluation**:
Comparing the ETIA Score for emotionally charged prompts against a neutral prompt to obtain a Differential ETIA Score, which reliably captures the impact of emotions on model behavior.

The pipeline is designed to work with various pre-trained, open-source LLMs, including Gemma-2, LLaMA-3.3, and Qwen-2.5, among others.

## Methodology

The framework follows these key steps:

- **Prompt Templates**:
Templates with a placeholder *{emotion}* are used as prompts. The *{emotion}* is substituted with emotion words *(e.g., anger, happiness, sadness, fear, surprise, disgust)* or with *“none”* as a neutral baseline.

- **Activation Extraction**:
During model inference, self-attention matrices are extracted from selected layers using the TransformerLens library, focusing on the attention mechanisms for the tokenized prompt.

- **ETIA Score Calculation**:
For each prompt, the attention weights directed toward emotion tokens are aggregated and normalized. The overall ETIA Score is derived by averaging these normalized values across chosen layers and heads.

- **Differential ETIA Score Evaluation**:
The score for an emotional prompt is compared with the neutral baseline prompt. A higher absolute Differential ETIA indicates a stronger emotional impact on the model's internal activations.

## Experiments

Experiments were conducted using 20 English prompts across **seven** emotions *(anger, disgust, fear, happiness, sadness, surprise, and none)* on 3 LLMs:

- `LLaMA-3.3-70B-instruct`
- `Qwen-2.5-72B-instruct`
- `Gemma-2-27B-it`

This systematic approach validates that the ETIA Framework can robustly measure and optimize emotional prompt effectiveness across different model architectures.

## License
This project is licensed under the Apache2.0 License.