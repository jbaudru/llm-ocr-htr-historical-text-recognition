import cv2
from PIL import Image
import numpy as np
import pandas as pd
import base64
import requests
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import easyocr
import pytesseract
import keras_ocr

from nltk.metrics.distance import jaccard_distance, masi_distance
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from Levenshtein import distance as levenshtein_distance


openai_API_KEY = "sk-proj-PwHJjpWHrxzyPJxQ8W0tT3BlbkFJk7rrTpTkpYZUI5L57Gf9"
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
# https://github.com/UB-Mannheim/tesseract/wiki

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def crop_image(image):
    #user32 = ctypes.windll.user32
    #screensize = user32.GetSystemMetrics(0)//2, user32.GetSystemMetrics(1)//2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
    blurred = cv2.GaussianBlur(gray, (35, 35), 0) # Apply Gaussian blur to the image to reduce noise
    alpha = 1.9 # Contrast control (1.0-3.0)
    beta = 20 # Brightness control (0-100)
    contrasted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
    edges = cv2.Canny(contrasted, 29, 29) # Use Canny edge detection to find edges in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 500, minLineLength=2000, maxLineGap=500) # Use Hough Line Transform to find lines in the image
    min_x, min_y = image.shape[1], image.shape[0]
    max_x, max_y = 0, 0
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Update the smallest and largest x and y values
            min_x, min_y = min(min_x, x1, x2), min(min_y, y1, y2)
            max_x, max_y = max(max_x, x1, x2), max(max_y, y1, y2)
        # Crop the image vertically and horizontally
        cropped = image[min_y:max_y, min_x:max_x]
        """
        image_tmp = cv2.resize(image, screensize)
        cv2.imshow('Image with Rectangle', image_tmp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        print("[INFO] Crop done!")
        return cropped
    except:
        print("[INFO] No crop possible!")
        return image

def color_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    return image

def easyOCR(image_path):
    reader = easyocr.Reader(['en'])
    img = cv2.imread(image_path)
    results = reader.readtext(img)
    output = []
    for res in results:
        #xy = res[0]
        #xy1, xy2, xy3, xy4 = xy[0], xy[1], xy[2], xy[3]
        det, conf = res[1], res[2]
        output.append((det, round(conf, 2))) 
    text = ' '.join([i[0] for i in output])
    return text


def pytesseractOCR(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text


def kerasOCR(image_path):
    pipeline = keras_ocr.pipeline.Pipeline()
    image = keras_ocr.tools.read(image_path)
    prediction_groups = pipeline.recognize([image])

    words = []
    for line in prediction_groups[0]:
        for word in line:
            try:
                if isinstance(word[0], str):
                    words.append(word[0])
            except IndexError:
                continue

    text = ' '.join(words)
    return text


def calamariOCR(image_path):
    pass

def askOpenAI(image_path):
    base64_image = encode_image(image_path)
    #prompt = "Give me the text in the image (a table), only the text that you are able to read, correct the text if there is missing information or typo, no additional information, no markdown element only '\n':"
    prompt = "Recognize the text in the image and correct it if necessary, the image contain a table and I want a table in .txt format as output, juste the table no other sentence from you:"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_API_KEY}"
    }    
    payload = {
        "model": "gpt-4o",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 1000,
        "temperature": 0
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


def save_text(text, output_path):
    with open(output_path.replace('.jpg', '.txt'), 'w', encoding='utf-8') as f:
        f.write(text)


def xlsx_to_string(filepath):
    df = pd.read_excel(filepath)
    string = df.to_string(index=False).replace("NaN", "").replace("\t", "")
    string = re.sub(' +', ' ', string)  # Replace multiple spaces with a single space
    string = string.replace("\n", " ")  # Replace line breaks with a space
    return string


def compute_distances(text1, text2):
    stop_words = set(stopwords.words('english')) 
    text1 = text1.translate(str.maketrans('', '', string.punctuation)).lower()
    text2 = text2.translate(str.maketrans('', '', string.punctuation)).lower()
    text1 = [i for i in word_tokenize(text1) if not i in stop_words]
    text2 = [i for i in word_tokenize(text2) if not i in stop_words]
    set1 = set(ngrams(text1, n=1))
    set2 = set(ngrams(text2, n=1))
    jaccard = jaccard_distance(set1, set2)
    masi = masi_distance(set1, set2)
    levenshtein = levenshtein_distance(text1, text2)
    return jaccard, masi, levenshtein


def compare_texts(texts, image_path):
    results = {name: [] for name in texts.keys()}
    for name1, t1 in texts.items():
        for name2, t2 in texts.items():
            jacc, mas, lev = compute_distances(t1, t2)
            results[name1].append((jacc, mas, lev))
    df_jacc = pd.DataFrame({name: [x[0] for x in res] for name, res in results.items()}, index=texts.keys())
    df_mas = pd.DataFrame({name: [x[1] for x in res] for name, res in results.items()}, index=texts.keys())
    df_lev = pd.DataFrame({name: [x[2] for x in res] for name, res in results.items()}, index=texts.keys())
    # Get the name of the image (without the extension)
    image_name = image_path.split('/')[-1].split('.')[0]
    # Save the dataframes to files
    df_jacc.to_csv(f'results/comparisons/{image_name}_jaccard.csv')
    df_mas.to_csv(f'results/comparisons/{image_name}_masi.csv')
    df_lev.to_csv(f'results/comparisons/{image_name}_levenshtein.csv')
    print("Jaccard distance:")
    print(df_jacc)
    print("\nMasi distance:")
    print(df_mas)
    print("\nLevenshtein distance:")
    print(df_lev)
    # Plot results in heatmaps
    for df, title in [(df_jacc, 'Jaccard'), (df_mas, 'Masi'), (df_lev, 'Levenshtein')]:
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
        plt.savefig(f'results/comparisons/{image_name}_{title.lower()}_heatmap.png')
        plt.tight_layout()
        plt.show()


def main():
    img_lst = ["data/Archives_LLN_Nivelles_I_1921_REG 5193/example1.jpeg", "data/Archives_LLN_Nivelles_I_1921_REG 5193/example2.jpeg"]
    trans_lst = ["data/transcriptions/transcription_ex1.xlsx", "data/transcriptions/transcription_ex2.xlsx"]
    for i in range(len(img_lst)):
        transcription = xlsx_to_string(trans_lst[i])
        image_path = img_lst[i]
        image = cv2.imread(image_path)
        
        croped_image = crop_image(image)
        cc_image = color_image(croped_image)
        
        output_path = image_path.replace('.jpeg', '_cc.jpeg')
        cv2.imwrite(output_path, cc_image)
        
        texts = {
            "Human" : transcription,
            "LLM" : askOpenAI(image_path),
            "LLM cc": askOpenAI(output_path),
            "EasyOCR": easyOCR(image_path),
            "EasyOCR cc": easyOCR(output_path),
            "Pytesseract": pytesseractOCR(image_path),
            "Pytesseract cc": pytesseractOCR(output_path),
            "KerasOCR": kerasOCR(image_path),
            "KerasOCR cc": kerasOCR(output_path),
        }
        compare_texts(texts, image_path)


if __name__ == '__main__':
    main()
    
# TODO: compare methods in terms of running time