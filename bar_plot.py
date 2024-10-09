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
    models = ["claude-3-5-sonnet-20240620", "gpt-4o", "EasyOCR", "KerasOCR", "Pytesseract"]
    
    bleu_scores = {exp: {model: [] for model in models} for exp in experiments}
    cer_scores = {exp: {model: [] for model in models} for exp in experiments}
    
    for exp in tqdm(experiments, desc=f"Processing experiments", leave=False, ascii=' >='):
        for model in tqdm(models, desc=f"Processing methods", leave=False, ascii=' >='):
            for i in tqdm(range(20), desc=f"Processing texts", leave=False, ascii=' >='):
                # TODO: Check that this is actually ignoring files
                if(exp=="one-example_prompt" and i==2):
                    #print("DEBUG: expertiment: ", exp, " file: ", i, "ignored")
                    continue
                elif(exp=="two-example_prompt" and (i==2 or i==3)):
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
    models = list(scores[experiments[0]].keys())
    
    # Calculate average BLEU scores for each model in each experiment
    avg_scores = {exp: {model: calculate_average_bleu(scores[exp][model]) for model in models} for exp in experiments}
    
    # Define other models to combine for comparison
    other_models = ["EasyOCR", "KerasOCR", "Pytesseract"]
    avg_scores_combined = {model: calculate_average_bleu([avg_scores[exp][model] for exp in experiments]) for model in other_models}
    
    # Prepare data for seaborn violin plot (flatten the dictionary into a long format)
    plot_data = []
    for exp in experiments:
        for model in ["gpt-4o", "claude-3-5-sonnet-20240620"]:
            for score in scores[exp][model]:
                plot_data.append([exp, model, score])
    
    for model in other_models:
        for score in scores[experiments[0]][model]:  # Assuming all experiments have the same set of models
            plot_data.append(["Combined", model, score])
    
    # Convert the data to a suitable format for seaborn
    plot_df = pd.DataFrame(plot_data, columns=["Experiment", "Model", "BLEU Score"])
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(18, 8))  # Increase figure size for better visibility
    
    # Create the violin plot
    sns.violinplot(x="Model", y="BLEU Score", hue="Experiment", data=plot_df, ax=ax, split=True)
    
    # Set labels and title
    ax.set_ylabel('BLEU Score')
    ax.set_title('BLEU Score Distribution by Model and Experiment')

    # Clean experiment names
    clean_experiment_names = [exp.replace('_', ' ').replace('-', ' ') for exp in experiments]

    # Set custom x-tick labels for better readability
    xtick_labels = ["gpt-4o", "claude-3-5-sonnet-20240620"] + other_models
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    
    # Add secondary x-axis for experiment names
    secax = ax.secondary_xaxis(-0.4)  # Move the secondary x-axis further down
    secax.set_xticks(np.arange(len(experiments)) * 2 + 0.5)
    secax.set_xticklabels(clean_experiment_names, rotation=0, ha="center")
    secax.set_xlim(-0.5, len(experiments) * 2 - 0.5)  # Set the limit to stop at the dotted vertical line
    
    # Show the plot
    fig.tight_layout()
    plt.grid(axis='y')
    plt.savefig("results/average-bleu_whole-scans_violin.png")
    plt.show()
    
    
    
    
def main():
    base_path = "results/postprocessed"
    base_path = "results/predictions"
    bleu_scores = collect_bleu_scores(base_path)
    plot_scores(bleu_scores)



if __name__ == "__main__":
    main()