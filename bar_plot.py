from lib.tools import Tools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

tools = Tools()

def calculate_average_bleu(scores):
    return np.mean(scores)

def collect_bleu_scores(base_path):
    experiments = ["zero-shot_simple-prompt", "zero-shot_complex-prompt", "one-example_prompt", "two-example_prompt", "refine_complex-prompt"]
    #models = ["claude-3-5-sonnet-20240620", "gpt-4o", "EasyOCR", "KerasOCR", "Pytesseract", "trOCR"]
    models = ["claude-3-5-sonnet-20240620", "gpt-4o", "EasyOCR", "Pytesseract", "KerasOCR", "trOCR"]
    
    bleu_scores = {exp: {model: [] for model in models} for exp in experiments}
    cer_scores = {exp: {model: [] for model in models} for exp in experiments}
    
    for exp in tqdm(experiments, desc=f"Processing experiments", leave=False, ascii=' >='):
        for model in tqdm(models, desc=f"Processing methods", leave=False, ascii=' >='):
            for i in tqdm(range(20), desc=f"Processing texts", leave=False, ascii=' >='):
                # TODO: Check that this is actually ignoring files
                if exp == "one-example_prompt" and i == 1:
                    continue
                if exp == "two-example_prompt" and (i == 1 or i == 2):
                    continue
                else:
                        try:
                            pred_path = base_path + "/" + exp + "/" + model + "/transcription" + str(i) + ".txt"
                            #print(pred_path)
                            with open(pred_path, "r", encoding="utf-8") as pred_file:
                                pred_text = pred_file.read()
                            #print(pred_text)
                            
                            gt_path = f"data/transcriptions/transcription_ex{i+1}.xlsx"
                            gt_text = tools.xlsx_to_string(gt_path)
                            
                            scores = tools.BLEU(pred_text, gt_text)
                            #print(scores)
                            bleu_score = scores
                            bleu_scores[exp][model].append(bleu_score)
                            #cer_score = scores[-3]
                            #cer_scores[exp][model].append(cer_score)
                        except:
                            print("Missing file: ", pred_path)

    return bleu_scores


def plot_scores(scores):
    experiments = list(scores.keys())
    
    # Define other models to combine for comparison
    llm_models = ["claude-3-5-sonnet-20240620", "gpt-4o"]
    other_models = ["EasyOCR", "KerasOCR", "Pytesseract", "trOCR"]

    # Prepare data for seaborn violin plot (flatten the dictionary into a long format)
    llm_data = []
    ocr_data = []

    # Collect LLM (GPT, Claude) data by experiment
    for exp in experiments:
        for model in llm_models:
            for score in scores[exp][model]:
                llm_data.append([exp, model, score])
    
    # Collect OCR data (combined across all experiments)
    for exp in experiments:
        for model in other_models:
            for score in scores[exp][model]:  # Assuming all experiments have the same set of models
                ocr_data.append([exp, model, score])
    
    # Convert the data to suitable format for seaborn
    llm_df = pd.DataFrame(llm_data, columns=["Methods", "Model", "BLEU Score"])
    ocr_df = pd.DataFrame(ocr_data, columns=["Methods", "Model", "BLEU Score"])
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(16, 8))  # Increase figure size for better visibility
    
    # Plot the LLM violin plots, grouped by experiment
    sns.violinplot(x="Methods", y="BLEU Score", hue="Model", data=llm_df, ax=ax, dodge=True, split=False, linewidth=0.75, width=0.75, palette=["#0ea982", "#cc9b7a"])
    
    # Add OCR models violin plot next to LLMs
    sns.violinplot(x="Model", y="BLEU Score", data=ocr_df, ax=ax, dodge=False, linewidth=0.75, width=0.75, palette=["#abdbe3", "#76b5c5", "#1e81b0", "#063970"])
    
    # Set labels
    ax.set_ylabel('BLEU Score')

    # Clean experiment names for x-ticks
    clean_experiment_names = [exp.replace('_', ' ').replace('-', ' ') for exp in experiments]

    # Set x-tick labels for experiments and models
    xtick_labels = clean_experiment_names + other_models
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    # Add vertical dotted line to separate LLMs from OCR models
    separation_x = len(experiments) - 0.5
    ax.axvline(separation_x, color='gray', linestyle='--')
    
    # Set y-axis to logit scale
    #ax.set_yscale('logit')
    ax.set_ylim(-0.1, 0.55) 
    #ax.set_yscale('symlog')
    #ax.set_ylim(-1, 1)

    # Adjust layout and show the plot
    fig.tight_layout()
    plt.grid(axis='y')
    plt.savefig("results/average-bleu_whole-scans.png", dpi=300)
    plt.show()
    
    
    
def main():
    base_path = "results/postprocessed"
    #base_path = "results/predictions"
    bleu_scores = collect_bleu_scores(base_path)
    plot_scores(bleu_scores)



if __name__ == "__main__":
    main()