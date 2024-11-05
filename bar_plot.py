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
    experiments = ["one-example_prompt", "two-example_prompt"]
    models = ["claude-3-5-sonnet-20240620", "gpt-4o"]
    
    bleu_scores = {exp: {model: [] for model in models} for exp in experiments}
    
    for exp in tqdm(experiments, desc=f"Processing experiments", leave=False, ascii=' >='):
        for model in tqdm(models, desc=f"Processing methods", leave=False, ascii=' >='):
            for i in tqdm(range(20), desc=f"Processing texts", leave=False, ascii=' >='):
                if exp == "one-example_prompt" and i == 1:
                    continue
                if exp == "two-example_prompt" and (i == 1 or i == 2):
                    continue
                else:
                        try:
                            pred_path = base_path + "/" + exp + "/" + model + "/new_transcription" + str(i) + ".txt"
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
                            
    for exp, models_dict in bleu_scores.items():
        for model, scores in models_dict.items():
            # Create DataFrame for each model in each experiment
            df = pd.DataFrame({model: scores})
            
            # Save DataFrame to CSV
            df.to_csv(f"results/scores_comparisons/eval_whole/BLEU_scores_{exp}_{model}.csv", index_label="Sample")
        
        
    return bleu_scores

def collect_cer_scores(base_path):
    experiments = ["one-example_prompt", "two-example_prompt"]
    models = ["claude-3-5-sonnet-20240620", "gpt-4o"]
    
    cer_scores = {exp: {model: [] for model in models} for exp in experiments}
    
    for exp in tqdm(experiments, desc=f"Processing experiments", leave=False, ascii=' >='):
        for model in tqdm(models, desc=f"Processing methods", leave=False, ascii=' >='):
            for i in tqdm(range(20), desc=f"Processing texts", leave=False, ascii=' >='):
                if exp == "one-example_prompt" and i == 1:
                    continue
                if exp == "two-example_prompt" and (i == 1 or i == 2):
                    continue
                else:
                    try:
                        pred_path = base_path + "/" + exp + "/" + model + "/new_transcription" + str(i) + ".txt"
                        with open(pred_path, "r", encoding="utf-8") as pred_file:
                            pred_text = pred_file.read()
                            
                        gt_path = f"data/transcriptions/transcription_ex{i+1}.xlsx"
                        gt_text = tools.xlsx_to_string(gt_path)
                        
                        scores = tools.CER(pred_text, gt_text)
                        cer_score = scores
                        cer_scores[exp][model].append(cer_score)
                    except:
                        print("Missing file: ", pred_path)
    
    for exp, models_dict in cer_scores.items():
        for model, scores in models_dict.items():
            # Create DataFrame for each model in each experiment
            df = pd.DataFrame({model: scores})
            
            # Save DataFrame to CSV
            df.to_csv(f"results/scores_comparisons/eval_whole/CER_scores_{exp}_{model}.csv", index_label="Sample")
        
    
    return cer_scores


def plot_scores(scores):
    experiments = list(scores.keys())
    
    # Define other models to combine for comparison
    llm_models = ["claude-3-5-sonnet-20240620", "gpt-4o"]
    other_models = ["EasyOCR", "KerasOCR", "Pytesseract", "trOCR"]
    tr_models = ["trOCR 20", "trOCR 50"]
    
    # Prepare data for seaborn violin plot (flatten the dictionary into a long format)
    llm_data = []
    ocr_data = []
    tr20_data = []
    tr50_data = []

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
                
    
    df = pd.read_csv('results/perline_predictions/bleu_TrOCR20_perline.csv')
    bleu_column = df['bleu']
    bleu_list = bleu_column.tolist()
    for score in bleu_list:
        tr20_data.append(["Fine-tunning", "trOCR 20", score])
        
    df = pd.read_csv('results/perline_predictions/bleu_TrOCR50_perline.csv')
    bleu_column = df['bleu']
    bleu_list = bleu_column.tolist()
    for score in bleu_list:
        tr50_data.append(["Fine-tunning", "trOCR 50", score])
    
    # Convert the data to suitable format for seaborn
    llm_df = pd.DataFrame(llm_data, columns=["Methods", "Model", "BLEU Score"])
    ocr_df = pd.DataFrame(ocr_data, columns=["Methods", "Model", "BLEU Score"])
    tr_df = pd.DataFrame(tr20_data + tr50_data, columns=["Methods", "Model", "BLEU Score"])
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(16, 8))  # Increase figure size for better visibility
    
    # Plot the LLM violin plots, grouped by experiment
    sns.violinplot(x="Methods", y="BLEU Score", hue="Model", data=llm_df, ax=ax, dodge=True, split=False, linewidth=0.75, width=0.75, palette=["#0ea982", "#cc9b7a"])
    
    # Add OCR models violin plot next to LLMs
    sns.violinplot(x="Model", y="BLEU Score", data=ocr_df, ax=ax, dodge=False, linewidth=0.75, width=0.75, palette=["#abdbe3", "#76b5c5", "#1e81b0", "#063970"])
    
    # Add TrOCR models violin plot next to OCR models
    sns.violinplot(x="Methods", y="BLEU Score", hue="Model", data=tr_df, ax=ax, dodge=True, linewidth=0.75, width=0.75, palette=["#fd75fd", "#8a44da"])
    
    # Set labels
    ax.set_ylabel('BLEU Score', fontsize=14)

    # Clean experiment names for x-ticks
    clean_experiment_names = [exp.replace('_', ' ').replace('-', ' ') for exp in experiments]

    # Set x-tick labels for experiments and models
    xtick_labels = clean_experiment_names + other_models + ["Fine-tunning"]
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    # Add vertical dotted line to separate LLMs from OCR models
    separation_x = len(experiments) - 0.5
    ax.axvline(separation_x, color='gray', linestyle='--')
    
    # add vertical dotted line to separate OCR models from TrOCR models
    separation_x = len(experiments) + len(other_models) - 0.5
    ax.axvline(separation_x, color='gray', linestyle='--')
    
    # Set y-axis to logit scale
    #ax.set_yscale('logit')
    ax.set_ylim(-0.1, 0.55) 
    #ax.set_yscale('symlog')
    #ax.set_ylim(-1, 1)

    # Set larger font size for tick labels
    ax.tick_params(axis='x', labelsize=14)
    
    # Adjust layout and show the plot
    fig.tight_layout()
    plt.grid(axis='y')
    plt.savefig("results/average-bleu.png", dpi=300)
    #plt.show()


def plot_scores_CER(scores):
    experiments = list(scores.keys())
    
    # Define other models to combine for comparison
    llm_models = ["claude-3-5-sonnet-20240620", "gpt-4o"]
    other_models = ["EasyOCR", "KerasOCR", "Pytesseract", "trOCR"]
    tr_models = ["trOCR 20", "trOCR 50"]
    
    # Prepare data for seaborn violin plot (flatten the dictionary into a long format)
    llm_data = []
    ocr_data = []
    tr20_data = []
    tr50_data = []

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
                
    
    df = pd.read_csv('results/perline_predictions/cer_TrOCR20_perline.csv')
    cer_column = df['cer']
    cer_list = cer_column.tolist()
    for score in cer_list:
        tr20_data.append(["Fine-tunning", "trOCR 20", score])
        
    df = pd.read_csv('results/perline_predictions/cer_TrOCR50_perline.csv')
    cer_column = df['cer']
    cer_list = cer_column.tolist()
    for score in cer_list:
        tr50_data.append(["Fine-tunning", "trOCR 50", score])
    
    # Convert the data to suitable format for seaborn
    llm_df = pd.DataFrame(llm_data, columns=["Methods", "Model", "CER"])
    ocr_df = pd.DataFrame(ocr_data, columns=["Methods", "Model", "CER"])
    tr_df = pd.DataFrame(tr20_data + tr50_data, columns=["Methods", "Model", "CER"])
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(16, 8))  # Increase figure size for better visibility
    
    # Plot the LLM violin plots, grouped by experiment
    sns.violinplot(x="Methods", y="CER", hue="Model", data=llm_df, ax=ax, dodge=True, split=False, linewidth=0.75, width=0.75, palette=["#0ea982", "#cc9b7a"])
    
    # Add OCR models violin plot next to LLMs
    sns.violinplot(x="Model", y="CER", data=ocr_df, ax=ax, dodge=False, linewidth=0.75, width=0.75, palette=["#abdbe3", "#76b5c5", "#1e81b0", "#063970"])
    
    # Add TrOCR models violin plot next to OCR models
    sns.violinplot(x="Methods", y="CER", hue="Model", data=tr_df, ax=ax, dodge=True, linewidth=0.75, width=0.75, palette=["#fd75fd", "#8a44da"])
    
    # Set labels
    ax.set_ylabel('CER', fontsize=14)

    # Clean experiment names for x-ticks
    clean_experiment_names = [exp.replace('_', ' ').replace('-', ' ') for exp in experiments]

    # Set x-tick labels for experiments and models
    xtick_labels = clean_experiment_names + other_models + ["Fine-tunning"]
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    # Add vertical dotted line to separate LLMs from OCR models
    separation_x = len(experiments) - 0.5
    ax.axvline(separation_x, color='gray', linestyle='--')
    
    # add vertical dotted line to separate OCR models from TrOCR models
    separation_x = len(experiments) + len(other_models) - 0.5
    ax.axvline(separation_x, color='gray', linestyle='--')
    
    # Set y-axis to logit scale
    #ax.set_yscale('logit')
    ax.set_ylim(0, 5) 
    #ax.set_yscale('symlog')
    #ax.set_ylim(-1, 1)
    
    # Set larger font size for tick labels
    ax.tick_params(axis='x', labelsize=14)

    # Adjust layout and show the plot
    fig.tight_layout()
    plt.grid(axis='y')
    plt.savefig("results/average-cer.png", dpi=300)
    #plt.show()
    
    
    
def main():
    base_path = "results/postprocessed"
    #base_path = "results/predictions"
    
    bleu_scores = collect_bleu_scores(base_path)
    
    #plot_scores(bleu_scores)
    
    cer_score = collect_cer_scores(base_path)
    #plot_scores_CER(cer_score)



if __name__ == "__main__":
    main()