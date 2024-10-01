from lib.tools import Tools
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

tools = Tools()

def calculate_average_bleu(scores):
    return np.mean(scores)

def collect_bleu_scores(base_path):
    experiments = ["zero-shot_simple-prompt", "zero-shot_complex-prompt", "one-example_prompt", "two-example_prompt", "refine_complex-prompt"]
    #models = ["claude-3-5-sonnet-20240620", "gpt-4o", "EasyOCR", "KerasOCR", "Pytesseract", "trOCR"]
    models = ["claude-3-5-sonnet-20240620", "gpt-4o", "EasyOCR", "KerasOCR", "Pytesseract"]
    
    bleu_scores = {exp: {model: [] for model in models} for exp in experiments}
    cer_scores = {exp: {model: [] for model in models} for exp in experiments}
    
    for exp in tqdm(experiments, desc=f"Processing experiments", leave=False, ascii=' >='):
        for model in tqdm(models, desc=f"Processing methods", leave=False, ascii=' >='):
            for i in tqdm(range(20), desc=f"Processing texts", leave=False, ascii=' >='):
                pred_path = base_path + "/" + exp + "/" + model + "/transcription" + str(i) + ".txt"
                
                with open(pred_path, "r", encoding="utf-8") as pred_file:
                    pred_text = pred_file.read()
                
                gt_path = f"data/transcriptions/transcription_ex{i+1}.xlsx"
                gt_text = tools.xlsx_to_string(gt_path)
                
                scores = tools.compute_distances(pred_text, gt_text)
                bleu_score = scores[-1]
                bleu_scores[exp][model].append(bleu_score)
                cer_score = scores[-3]
                cer_scores[exp][model].append(cer_score)
            
    return bleu_scores


def plot_scores(scores):
    experiments = list(scores.keys())
    models = list(scores[experiments[0]].keys())
    
    avg_scores = {exp: {model: calculate_average_bleu(scores[exp][model]) for model in models} for exp in experiments}
    
    # Calculate average BLEU scores for non-GPT and non-Sonnet models across all experiments
    #other_models = ["EasyOCR", "KerasOCR", "Pytesseract", "trOCR"]
    other_models = ["EasyOCR", "KerasOCR", "Pytesseract"]
    avg_scores_combined = {model: calculate_average_bleu([avg_scores[exp][model] for exp in experiments]) for model in other_models}
    
    x = np.arange(len(experiments) * 2 + len(other_models))  # 5*2 for GPT and Sonnet, 4 for OCR methods
    width = 0.35

    fig, ax = plt.subplots(figsize=(18, 8))  # Increase figure size for a longer plot

    # Plot GPT and Sonnet models for each experiment
    colors = ['blue', 'orange']  # Define colors for GPT and Sonnet
    for idx, exp in enumerate(experiments):
        for model_idx, model in enumerate(["gpt-4o", "claude-3-5-sonnet-20240620"]):
            model_scores = avg_scores[exp][model]
            ax.bar(idx * 2 + model_idx * 3*width, model_scores, width, label=f"{model}" if idx == 0 else "", color=colors[model_idx])

    # Plot combined OCR methods
    for idx, model in enumerate(other_models):
        model_scores = avg_scores_combined[model]
        ax.bar(len(experiments) * 2 + idx, model_scores, width, label=model)

    ax.set_ylabel('Average BLEU Score')
    ax.set_title('Average BLEU Score by Method')

    # Clean experiment names
    clean_experiment_names = [exp.replace('_', ' ').replace('-', ' ') for exp in experiments]

    # Create x-tick labels
    xtick_labels = []
    for exp in clean_experiment_names:
        xtick_labels.append("gpt-4o")
        xtick_labels.append("claude-3-5-sonnet-20240620")
    xtick_labels.extend(other_models)

    # Create secondary x-tick labels for experiment names
    exp_labels = []
    for exp in clean_experiment_names:
        exp_labels.append(exp)
        exp_labels.append("")
    exp_labels.extend([""] * len(other_models))

    ax.set_xticks(np.arange(len(experiments) * 2 + len(other_models)))
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    # Add secondary x-axis for experiment names
    secax = ax.secondary_xaxis(-0.4)  # Move the secondary x-axis further down
    secax.set_xticks(np.arange(len(experiments)) * 2 + 0.5)
    secax.set_xticklabels(clean_experiment_names, rotation=0, ha="center")
    secax.set_xlim(-0.5, len(experiments) * 2 - 0.5)  # Set the limit to stop at the dotted vertical line

    ax.legend()

    # Add vertical dotted line to separate GPT and Sonnet models from OCR methods
    separation_x = len(experiments) * 2 - 0.5
    ax.axvline(separation_x, color='gray', linestyle='--')

    fig.tight_layout()
    plt.savefig("results/average-bleu_whole-scans.png")
    plt.show()
    
    
    
    
def main():
    base_path = "results/predictions"
    bleu_scores = collect_bleu_scores(base_path)
    plot_scores(bleu_scores)



if __name__ == "__main__":
    main()