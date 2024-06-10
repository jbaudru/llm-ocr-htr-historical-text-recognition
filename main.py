import cv2
import numpy as np
import base64
import requests

import easyocr
import matplotlib.pyplot as plt

import nltk
from nltk.metrics.distance import jaccard_distance, masi_distance
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from Levenshtein import distance as levenshtein_distance

api_key = "sk-proj-PwHJjpWHrxzyPJxQ8W0tT3BlbkFJk7rrTpTkpYZUI5L57Gf9"

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

def easyOCR(image_path):
    # This needs to run only once to load the model into memory
    reader = easyocr.Reader(['en'])

    # reading the image
    img = cv2.imread(image_path)

    # run OCR
    results = reader.readtext(img)

    # show the image and plot the results
    plt.imshow(img)
    output = []
    for res in results:
        # bbox coordinates of the detected text
        xy = res[0]
        xy1, xy2, xy3, xy4 = xy[0], xy[1], xy[2], xy[3]
        # text results and confidence of detection
        det, conf = res[1], res[2]
        # show time :)
        plt.plot([xy1[0], xy2[0], xy3[0], xy4[0], xy1[0]], [xy1[1], xy2[1], xy3[1], xy4[1], xy1[1]], 'r-')
        plt.text(xy1[0], xy1[1], f'{det} [{round(conf, 2)}]')   
        output.append((det, round(conf, 2))) 
    return output
    
def askOpenAI(image_path):
    base64_image = encode_image(image_path)

    #prompt = "Give me the text in the image (a table), only the text that you are able to read, correct the text if there is missing information or typo, no additional information, no markdown element only '\n':"
    prompt = "Recognize the text in the image and correct it if necessary, the image contain a table and I want a table in .txt format as output, juste the table no other sentence from you:"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
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


def main():
    img_lst = ['data/img_0.jpg', 'data/img_1.jpg']
    for image_path in img_lst:
        
        print("[INFO] Processing image: ", image_path)
        image = cv2.imread(image_path)
        croped_image = crop_image(image)
        output_path = image_path.replace('.jpg', '_cropped.jpg')
        cv2.imwrite(output_path, croped_image)
        
        text = askOpenAI(output_path)
        text2 = easyOCR(output_path)
        
        jacc, mas, lev = compute_distances(text, text2)
        print(f"Jaccard distance: {jacc}, Masi distance: {mas}, Levenshtein distance: {lev}")
        
        #print("[INFO] Text found.")
        #save_text(text, output_path)
        

if __name__ == '__main__':
    main()
    
    
# TODO: Report
# TODO: Compare distance and the text here VS the text found by classical OCR software VS real text readed by human 