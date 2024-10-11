import pandas as pd
import re
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import json
import os

import nltk 
from nltk.metrics.distance import jaccard_distance, masi_distance
import evaluate
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')

import string
from Levenshtein import distance as levenshtein_distance
from torchmetrics.text import CharErrorRate
import jiwer

import requests
from bs4 import BeautifulSoup

class Tools:
    def save_text(self, text, output_path):
        with open(output_path.replace('.jpg', '.txt'), 'w', encoding='utf-8') as f:
            f.write(text)

    def convert_excel_to_json(self, file_path, output_folder):
        # Read the Excel file with two header rows -- Modify this if your Excel file has a different structure
        excel_data = pd.read_excel(file_path, header=[0, 1])

        # Convert the DataFrame to a nested dictionary
        nested_dict = {}
        for idx, row in excel_data.iterrows():
            nested_row = {}
            for col in excel_data.columns:
                header1, header2 = col
                if header1 not in nested_row:
                    nested_row[header1] = {}
                nested_row[header1][header2] = row[col]
            nested_dict[idx] = nested_row

        # Convert the nested dictionary to JSON
        json_data = json.dumps(list(nested_dict.values()), indent=4, ensure_ascii=False)
        
        # Extract file name without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        json_file_path = os.path.join(output_folder, f"{file_name}.json")
        
        # Save JSON data to a file
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json_data)

        print(f"Conversion complete. JSON data saved to '{json_file_path}'.")

        """
        output_folder = os.path.join(os.getcwd(), 'data/json_transcriptions')

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        

        # Process each Excel file in the folder
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.xlsx'):
                file_path = os.path.join(input_folder, file_name)
                convert_excel_to_json(file_path, output_folder)

        print("All files have been processed.")
        """

    def xlsx_to_string(self, filepath):
        df = pd.read_excel(filepath)
        
        # Drop rows that don't have any information in any columns
        df = df.dropna(how='all')
        
        # Fill NaN values with a whitespace
        df = df.fillna(" ")
        
        string = df.to_string(index=False, header=False)
        string = re.sub(' +', ' ', string)  # Replace multiple spaces with a single space
        string = string.replace("\n", " \n")  # Ensure each new row starts on a new line
        return string
    
    
    def getData(self):
        # Open the text file in append mode
        with open("data_rag/names.txt", "a") as file:
            # Send a HTTP request to the webpage
            for i in range(1, 50): # the 50 pages
                response = requests.get("https://nl.geneanet.org/genealogie/?page=" + str(i))
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    li_elements = soup.select("div.panel li")
                    for li in li_elements:
                        name = li.get_text(strip=True).split("(")[0]
                        # Write the name to the text file
                        file.write(name + "\n")
                else:
                    print(f"Failed to get the webpage: {response.status_code}")

    def compute_distances(self, pred, gt):
        CER = CharErrorRate()
        """
        stop_words = set(stopwords.words('french')) 
        pred = pred.translate(str.maketrans('', '', string.punctuation)).lower()
        gt = gt.translate(str.maketrans('', '', string.punctuation)).lower()
        pred = [i for i in word_tokenize(pred) if not i in stop_words]
        gt = [i for i in word_tokenize(gt) if not i in stop_words]
        """
        #gt = gt.split("\n")
        #pred = pred.split("\n")
        set1 = set(ngrams(pred, n=1))
        set2 = set(ngrams(gt, n=1))

        try:
            jaccard = jaccard_distance(set1, set2)
        except:
            jaccard = 0
            
        try:
            masi = masi_distance(set1, set2)
        except:
            masi = 0
            
        try:
            levenshtein = levenshtein_distance(pred, gt)
        except:
            levenshtein = 0
            
        try:
            cer = CER(pred, gt).item()
        except:
            cer = 0
            
        try:
            wer = jiwer.wer(pred, gt)
        except:
            wer = 0
            
        try:
            if len(pred) == 0:
                bleu = 0
            else:
                bleu_metric = evaluate.load("bleu")
                bleu = bleu_metric.compute(predictions=[pred], references=[[gt]])['bleu']
        except:
            bleu = 0
            
        return jaccard, masi, levenshtein, cer, bleu, wer
    
    def BLEU(self, pred, gt):
        if len(pred) == 0:
            return 0
        else:
            bleu_metric = evaluate.load("bleu")
            bleu = bleu_metric.compute(predictions=[pred], references=[[gt]])['bleu']
            return bleu
        
    def CER(self, pred, gt):
        CER = CharErrorRate()
        cer = CER(pred, gt).item()
        return cer
        
    def CERreduction(self, prev_pred, curr_pred, gt):
        CERred = (self.CER(prev_pred, gt) - self.CER(curr_pred, gt)) / self.CER(prev_pred, gt)
        return CERred
    
    def PCIS(self, prev_pred, curr_pred, gt):
        lev_origsim = levenshtein_distance(prev_pred, gt)
        lev_llmsim = levenshtein_distance(curr_pred, gt)
        if(lev_origsim == 0):
            return min(max(lev_llmsim, -1), 1)
        elif(lev_llmsim != lev_origsim):
            return min(max((lev_llmsim-lev_origsim)/lev_origsim, -1), 1)
        else:
            return 0
    

    def compare_texts_violin_plot(self, texts, image_name):
        cer_dict = {
            "gpt-4o": [],
            "claude-3-5-sonnet-20240620": [],
            "EasyOCR": [],
            "Pytesseract": [],
            "KerasOCR": [],
            "trOCR": [],
        }
        
        blue_dict = {
            "gpt-4o": [],
            "claude-3-5-sonnet-20240620": [],
            "EasyOCR": [],
            "Pytesseract": [],
            "KerasOCR": [],
            "trOCR": [],
        }
        
        wer_dict = {
            "gpt-4o": [],
            "claude-3-5-sonnet-20240620": [],
            "EasyOCR": [],
            "Pytesseract": [],
            "KerasOCR": [],
            "trOCR": [],
        }
        
        models = list(cer_dict.keys())
        
        for model in texts:
            if model != "GT":
                for i in range(len(texts[model])):
                    gd = texts["GT"][i]
                    pred = texts[model][i]
                    jacc, mas, lev, cer, blue, wer = self.compute_distances(pred, gd)
                    cer_dict[model].append(cer)
                    blue_dict[model].append(blue)
                    wer_dict[model].append(wer)
        # CER
        data = []
        for model in models:
            for cer in cer_dict[model]:
                data.append((model, cer))
        df = pd.DataFrame(data, columns=["Model", "CER"])
        df.to_csv("results/comparisons/" + image_name + "_cer.csv")
        plt.figure(figsize=(12, 12))
        sns.violinplot(x="Model", y="CER", data=df, palette="Set3")
        plt.title('Character Error Rate (CER)')
        plt.xlabel('Models')
        plt.ylabel('CER')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y')
        plt.savefig("results/comparisons/" + image_name + "_violinplot_cer.png")
        plt.show()
        
        # BLEU
        data_blue = []
        for model in models:
            for blue in blue_dict[model]:
                data_blue.append((model, blue))
        df2 = pd.DataFrame(data_blue, columns=["Model", "BLEU"])
        df2.to_csv("results/comparisons/" + image_name + "_bleu.csv")
        plt.figure(figsize=(12, 12))
        sns.violinplot(x="Model", y="BLEU", data=df2, palette="Set3")
        plt.title('BLEU Score')
        plt.xlabel('Models')
        plt.ylabel('BLEU')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y')
        plt.savefig("results/comparisons/" + image_name + "_violinplot_bleu.png")
        plt.show()
        
        # WER
        data_wer = []
        for model in models:
            for wer in wer_dict[model]:
                data_wer.append((model, wer))
        df3 = pd.DataFrame(data_wer, columns=["Model", "WER"])
        df3.to_csv("results/comparisons/" + image_name + "_wer.csv")
        plt.figure(figsize=(12, 12))
        sns.violinplot(x="Model", y="WER", data=df3, palette="Set3")
        plt.title('Word Error Rate (WER)')
        plt.xlabel('Models')
        plt.ylabel('WER')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y')
        plt.savefig("results/comparisons/" + image_name + "_violinplot_wer.png")
        plt.show()

    def bar_plot(self, texts, image_name):
        #create a bar for each model with BLUE average
        blue_dict = {
            "gpt-4o": [],
            "claude-3-5-sonnet-20240620": [],
            "EasyOCR": [],
            "Pytesseract": [],
            "KerasOCR": [],
            "trOCR": [],
        }
        
        models = list(blue_dict.keys())
        
        for model in texts:
            if model != "GT":
                for i in range(len(texts[model])):
                    gd = texts["GT"][i]
                    pred = texts[model][i]
                    jacc, mas, lev, cer, blue, wer = self.compute_distances(pred, gd)
                    blue_dict[model].append(blue)
        
        data_blue = []
        for model in models:
            blue_avg = sum(blue_dict[model]) / len(blue_dict[model])
            data_blue.append((model, blue_avg))
        
        df2 = pd.DataFrame(data_blue, columns=["Model", "BLEU"])
        plt.figure(figsize=(12, 12))      
        sns.barplot(x="Model", y="BLEU", data=df2, palette="Set3")
        plt.title('BLEU Score')
        plt.xlabel('Models')
        plt.ylabel('BLEU')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y')
        plt.savefig("results/comparisons/" + image_name + "_barplot_bleu.png")
        plt.show()
                            

    def compare_texts(self, texts, image_name):
        results = {name: [] for name in texts.keys()}
        
        for name1, t1 in texts.items():
            for name2, t2 in texts.items():
                for k in range(len(t1)):
                    text1 = t1[k]
                    text2 = t2[k]
                    jacc, mas, lev, cer = self.compute_distances(text1, text2)
                    results[name1].append((jacc, mas, lev, cer))
                    jacc, mas, lev, cer = self.compute_distances(text2, text1)
                    results[name2].append((jacc, mas, lev, cer))
        
        results_averages = {}
        for name, res in results.items():
            # Initialize sums for each metric
            jacc_sum, mas_sum, lev_sum, cer_sum = 0, 0, 0, 0
            # Initialize count for non-NaN CER values
            cer_count = 0
            for jacc, mas, lev, cer in res:
                jacc_sum += jacc
                mas_sum += mas
                lev_sum += lev
                # Check if CER is not NaN before adding to sum and incrementing count
                if not math.isnan(cer):
                    cer_sum += cer
                    cer_count += 1
            # Compute averages, for CER use cer_count to avoid division by zero
            jacc_avg = jacc_sum / len(res)
            mas_avg = mas_sum / len(res)
            lev_avg = lev_sum / len(res)
            cer_avg = cer_sum / cer_count if cer_count else float('nan')
            results_averages[name] = (jacc_avg, mas_avg, lev_avg, cer_avg)
        
        
        df_jacc = pd.DataFrame({name: [results_averages[name][0]] for name in results_averages.keys()}, index=texts.keys())
        df_mas = pd.DataFrame({name: [results_averages[name][1]] for name in results_averages.keys()}, index=texts.keys())
        df_lev = pd.DataFrame({name: [results_averages[name][2]] for name in results_averages.keys()}, index=texts.keys())
        df_cer = pd.DataFrame({name: [results_averages[name][3]] for name in results_averages.keys()}, index=texts.keys())

        # Get the name of the image (without the extension)
        image_name = image_name.split('/')[-1].split('.')[0]
        # Save the dataframes to files
        df_jacc.to_csv("results/comparisons/" + image_name + "_jaccard.csv")
        df_mas.to_csv("results/comparisons/" + image_name + "_masi.csv")
        df_lev.to_csv("results/comparisons/" + image_name + "_levenshtein.csv")
        df_cer.to_csv("results/comparisons/" + image_name + "_cer.csv")

        # Plot results in heatmaps
        for df, title in [(df_jacc, 'Jaccard'), (df_mas, 'Masi'), (df_lev, 'Levenshtein'), (df_cer, 'Character Error Rate')]:
            plt.figure(figsize=(8, 8))
            ax = sns.heatmap(df, annot=True, fmt=".2f", annot_kws={"size": 5}, cmap='coolwarm')
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            plt.xticks(rotation=90)
            ax.set_aspect('equal')
            rect = patches.Rectangle((0,0), len(df.columns), 1, linewidth=2, edgecolor='purple', facecolor='none')
            rect2 = patches.Rectangle((0,0), 1, len(df.index), linewidth=2, edgecolor='purple', facecolor='none')
            ax.add_patch(rect)
            ax.add_patch(rect2)
            plt.title(f'{title} Distance Heatmap', y=-0.1)
            plt.savefig("results/comparisons/" + image_name + "_" + title.lower() + "_heatmap.png")
            plt.tight_layout()
            plt.show()
