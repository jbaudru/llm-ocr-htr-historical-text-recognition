from lib.tools import Tools
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

tools = Tools()

def calculate_average_bleu(scores):
    return np.mean(scores)

def collect_bleu_scores(base_path):
    experiments = ["zero-shot_complex-prompt", "zero-shot_simple-prompt", "one-example_prompt", "two-example_prompt", "refine_complex-prompt"]
    models = ["claude-3-5-sonnet-20240620", "gpt-4o", "EasyOCR", "KerasOCR", "Pytesseract", "trOCR"]
    
    bleu_scores = {exp: {model: [] for model in models} for exp in experiments}
    
    for exp in experiments:
        for model in tqdm(models, desc=f"Processing {exp}", ascii=' >='):
            for i in range(20):
                pred_path = base_path + "/" + exp + "/" + model + "/transcription" + str(i) + ".txt"
                
                with open(pred_path, "r", encoding="utf-8") as pred_file:
                    pred_text = pred_file.read()
                
                gt_path = f"data/transcriptions/transcription_ex{i+1}.xlsx"
                gt_text = tools.xlsx_to_string(gt_path)
                
                
                bleu_score = tools.compute_distances(pred_text, gt_text)[-1]
                bleu_scores[exp][model].append(bleu_score)
    
    return bleu_scores


def plot_bleu_scores(bleu_scores):
    experiments = list(bleu_scores.keys())
    models = list(bleu_scores[experiments[0]].keys())
    
    avg_bleu_scores = {exp: {model: calculate_average_bleu(bleu_scores[exp][model]) for model in models} for exp in experiments}
    
    # Calculate average BLEU scores for non-GPT and non-Sonnet models across all experiments
    other_models = ["EasyOCR", "KerasOCR", "Pytesseract", "trOCR"]
    avg_bleu_scores_combined = {model: calculate_average_bleu([avg_bleu_scores[exp][model] for exp in experiments]) for model in other_models}
    
    x = np.arange(len(experiments) + 1)  # +1 for the combined average of other models
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(18, 10))  # Increase figure size for a longer plot
    
    for idx, model in enumerate(models):
        if model in other_models:
            model_scores = [avg_bleu_scores_combined[model]] * (len(experiments) + 1)
        else:
            model_scores = [avg_bleu_scores[exp][model] for exp in experiments] + [0]  # Add 0 for the combined average position
        
        ax.bar(x + idx * width, model_scores, width, label=model)
    
    ax.set_ylabel('Average BLEU Score')
    ax.set_title('Average BLEU Score by Model and Experiment')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(experiments + ["Combined Average"])
    ax.legend()
    
    # Add vertical dotted line to separate "gpt-4o" and "claude-3-5-sonnet-20240620" from others
    separation_idx = 2  # Index after "gpt-4o" and "claude-3-5-sonnet-20240620"
    separation_x = x + separation_idx * width - width / 2
    for sep_x in separation_x:
        ax.axvline(sep_x, color='gray', linestyle='--')
    
    fig.tight_layout()
    plt.savefig("results/average_bleu_scores.png")
    plt.show()
    
    
def main():
    base_path = "results/predictions"
    bleu_scores = collect_bleu_scores(base_path)
    plot_bleu_scores(bleu_scores)

if __name__ == "__main__":
    main()