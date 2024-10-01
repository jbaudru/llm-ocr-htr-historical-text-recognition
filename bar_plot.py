from lib.tools import Tools
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

tools = Tools()

def calculate_average_bleu(scores):
    return np.mean(scores)

def collect_bleu_scores(base_path):
    experiments = ["zero-shot_complex-prompt", "zero-shot_simple-prompt", "one-example-prompt", "two-example-prompt", "refine-prompt"]
    models = ["claude-3-5-sonnet-20240620", "gpt-4o", "EasyOCR", "KerasOCR", "Pytesseract", "trOCR"]
    
    bleu_scores = {exp: {model: [] for model in models} for exp in experiments}
    
    for exp in experiments:
        for model in models:
            for i in range(20):
                pred_path = os.path.join(base_path, exp, model, f"transcription{i}")
                gt_path = f"data/transcriptions/transcription_ex{i+1}.txt"
                
                with open(gt_path, "r", encoding="utf-8") as gt_file:
                    gt_text = gt_file.read()
                
                with open(pred_path, "r", encoding="utf-8") as pred_file:
                    pred_text = pred_file.read()
                
                bleu_score = tools.compute_distances(pred_text, gt_text)[-1]
                bleu_scores[exp][model].append(bleu_score)
    
    return bleu_scores

def plot_bleu_scores(bleu_scores):
    experiments = list(bleu_scores.keys())
    models = list(bleu_scores[experiments[0]].keys())
    
    avg_bleu_scores = {exp: {model: calculate_average_bleu(bleu_scores[exp][model]) for model in models} for exp in experiments}
    
    x = np.arange(len(experiments))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, model in enumerate(models):
        model_scores = [avg_bleu_scores[exp][model] for exp in experiments]
        ax.bar(x + idx * width, model_scores, width, label=model)
    
    ax.set_ylabel('Average BLEU Score')
    ax.set_title('Average BLEU Score by Model and Experiment')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(experiments)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig("results/average_bleu_scores.png")
    plt.show()

def process_data():
    gt_lst = []
    pred_lst = []
    for i in tqdm(range(0, 20), ascii=' >='): #20 max
        i += 1
        trans = "data/transcriptions/transcription_ex" + str(i) + ".xlsx"
        trans_txt = tools.xlsx_to_string(trans)
        # save transcription into a text file
        with open(f"data/transcriptions/transcription_ex{i}.txt", "w", encoding="utf-8") as f:
            f.write(trans_txt)
        gt_lst.append(trans_txt)
    
def main():
    base_path = "results/predictions"
    bleu_scores = collect_bleu_scores(base_path)
    plot_bleu_scores(bleu_scores)

if __name__ == "__main__":
    main()