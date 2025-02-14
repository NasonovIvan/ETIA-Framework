### Entry point: loads the model, reads the configuration, and runs evaluations ###

import torch
from transformer_lens import HookedTransformer
from huggingface_hub import login

from config import PROMPTS, EMOTIONS
from src.evaluation import evaluate_emotional_prompts
from dotenv import load_dotenv
import os

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Login to HuggingFace if needed (replace with your token)
login(token=os.getenv("HUGGINGFACE_API_TOKEN"))

# Load the model
model = HookedTransformer.from_pretrained("gpt2", device=device)

# Run evaluation
if __name__ == "__main__":
    results = evaluate_emotional_prompts(model, PROMPTS, EMOTIONS)
