### Contains the function to evaluate emotional prompts and save results to CSV files ###

import os
import csv
from datetime import datetime
from src.etias_metrics import compute_etias, compute_diff_etias, compute_diff_detection


def evaluate_emotional_prompts(
    model, prompts: dict, emotions: list, output_dir="results"
):
    """
    Calculates all metrics for each prompt and each emotion, as well as the average values of metrics for all emotions.
    Saves the results to CSV files.

    Args:
        model: LLM model.
        prompts (dict): A dictionary of industrial patterns with key names.
        emotions (list): A list of emotions to analyze.
        output_dir (str): The directory for saving the results.

    Returns:
        dict: Results in the form of a dictionary with metrics for each prompt and each emotion,
              including the average values of metrics for all emotions.
    """

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {}

    for prompt_name, prompt_template in prompts.items():
        csv_filename = os.path.join(
            output_dir, f"{prompt_name}_{timestamp}_results.csv"
        )

        results[prompt_name] = {}
        print(f"\nPrompt: {prompt_name}")

        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)

            csv_writer.writerow(
                [
                    "Prompt",
                    "Emotion",
                    "ETIAS",
                    "Diff_ETIAS",
                    "Diff_Detection",
                    "Error (if any)",
                ]
            )

            etias_scores = []
            diff_etias_scores = []
            diff_detection_scores = []

            for emotion in emotions:
                try:
                    etias_score = compute_etias(
                        model, prompt_template.format(emotion=emotion), emotion
                    )
                    diff_etias_score = compute_diff_etias(
                        model, prompt_template, emotion
                    )

                    diff_detection_score = compute_diff_detection(
                        model, prompt_template, emotion
                    )  # THIS METRIC DOES NOT USE IN THE EMOTIONAL PROMPTING ANALYSIS

                    results[prompt_name][emotion] = {
                        "ETIAS": etias_score,
                        "Diff_ETIAS": diff_etias_score,
                        "Diff_Detection": diff_detection_score,
                    }

                    csv_writer.writerow(
                        [
                            prompt_name,
                            emotion,
                            f"{etias_score:.4f}",
                            f"{diff_etias_score:.4f}",
                            f"{diff_detection_score:.4f}",
                            "",
                        ]
                    )

                    etias_scores.append(etias_score)
                    diff_etias_scores.append(diff_etias_score)
                    diff_detection_scores.append(diff_detection_score)

                    print(
                        f"  Эмоция: {emotion} | "
                        f"ETIAS: {etias_score:.4f} | "
                        f"Diff_ETIAS: {diff_etias_score:.4f} | "
                        f"Diff_Detection: {diff_detection_score:.4f}"
                    )

                except Exception as e:
                    error_msg = str(e)
                    print(f"  Error for emotion '{emotion}': {error_msg}")

                    csv_writer.writerow([prompt_name, emotion, "", "", "", error_msg])

                    results[prompt_name][emotion] = {"error": error_msg}

            if etias_scores and diff_etias_scores and diff_detection_scores:
                average_etias = sum(etias_scores) / len(etias_scores)
                average_diff_etias = sum(diff_etias_scores) / len(diff_etias_scores)
                average_diff_detection = sum(diff_detection_scores) / len(
                    diff_detection_scores
                )

                results[prompt_name]["average"] = {
                    "ETIAS": average_etias,
                    "Diff_ETIAS": average_diff_etias,
                    "Diff_Detection": average_diff_detection,
                }

                csv_writer.writerow(
                    [
                        prompt_name,
                        "AVERAGE",
                        f"{average_etias:.4f}",
                        f"{average_diff_etias:.4f}",
                        f"{average_diff_detection:.4f}",
                        "",
                    ]
                )

                print(
                    f"  Average values for all emotions | "
                    f"ETIAS: {average_etias:.4f} | "
                    f"Diff_ETIAS: {average_diff_etias:.4f} | "
                    f"Diff_Detection: {average_diff_detection:.4f}"
                )

        print(f"The results are saved in {csv_filename}")

    return results
