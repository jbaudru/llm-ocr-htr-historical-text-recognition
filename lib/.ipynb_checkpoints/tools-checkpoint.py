import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from nltk.metrics.distance import jaccard_distance, masi_distance
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from Levenshtein import distance as levenshtein_distance
from torchmetrics.text import CharErrorRate

import requests
from bs4 import BeautifulSoup

class Tools:
    def save_text(self, text, output_path):
        with open(output_path.replace('.jpg', '.txt'), 'w', encoding='utf-8') as f:
            f.write(text)


    def xlsx_to_string(self, filepath):
        df = pd.read_excel(filepath)
        string = df.to_string(index=False).replace("NaN", "").replace("\t", "")
        string = re.sub(' +', ' ', string)  # Replace multiple spaces with a single space
        string = string.replace("\n", " ")  # Replace line breaks with a space
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

    def compute_distances(self, text1, text2):
        CER = CharErrorRate()
        stop_words = set(stopwords.words('french')) 
        text1 = text1.translate(str.maketrans('', '', string.punctuation)).lower()
        text2 = text2.translate(str.maketrans('', '', string.punctuation)).lower()
        text1 = [i for i in word_tokenize(text1) if not i in stop_words]
        text2 = [i for i in word_tokenize(text2) if not i in stop_words]
        set1 = set(ngrams(text1, n=1))
        set2 = set(ngrams(text2, n=1))
        jaccard = jaccard_distance(set1, set2)
        masi = masi_distance(set1, set2)
        levenshtein = levenshtein_distance(text1, text2)
        cer = CER(text1, text2).item()
        return jaccard, masi, levenshtein, cer


    def compare_texts(self, texts, image_path):
        results = {name: [] for name in texts.keys()}
        for name1, t1 in texts.items():
            for name2, t2 in texts.items():
                jacc, mas, lev, cer = self.compute_distances(t1, t2)
                results[name1].append((jacc, mas, lev, cer))
                
        df_jacc = pd.DataFrame({name: [x[0] for x in res] for name, res in results.items()}, index=texts.keys())
        df_mas = pd.DataFrame({name: [x[1] for x in res] for name, res in results.items()}, index=texts.keys())
        df_lev = pd.DataFrame({name: [x[2] for x in res] for name, res in results.items()}, index=texts.keys())
        df_cer = pd.DataFrame({name: [x[3] for x in res] for name, res in results.items()}, index=texts.keys())
        # Get the name of the image (without the extension)
        image_name = image_path.split('/')[-1].split('.')[0]
        # Save the dataframes to files
        df_jacc.to_csv("results/comparisons/" + image_name + "_jaccard.csv")
        df_mas.to_csv("results/comparisons/" + image_name + "_masi.csv")
        df_lev.to_csv("results/comparisons/" + image_name + "_levenshtein.csv")
        df_cer.to_csv("results/comparisons/" + image_name + "_cer.csv")

        # Plot results in heatmaps
        for df, title in [(df_jacc, 'Jaccard'), (df_mas, 'Masi'), (df_lev, 'Levenshtein'), (df_cer, 'Character Error Rate')]:
            plt.figure(figsize=(8, 8))
            ax = sns.heatmap(df, annot=True, fmt=".2f", annot_kws={"size": 10}, cmap='coolwarm')
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
