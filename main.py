import cv2
import numpy as np
import base64
import requests

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
        print("[INFO] No crop needed")
        return image
    
    
def askOpenAI(image_path):
    # Getting the base64 string
    base64_image = encode_image(image_path)

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
                "text": "Give me the text in the image (a table), only the text that you are able to read, correct the text if there is missing information or typo, no additional information, no markdown element only '\n':"
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
    return response.json()

def main():
    # Load the image
    img_lst = ['data/img_0.jpg', 'data/img_1.jpg']
    for image_path in img_lst:
        print("[INFO] Processing image: ", image_path)
        image = cv2.imread(image_path)
        croped_image = crop_image(image)
        output_path = image_path.replace('.jpg', '_cropped.jpg')
        cv2.imwrite(output_path, croped_image)
        text_json = askOpenAI(output_path)
        text = text_json["choices"][0]["message"]["content"]
        
        print("[INFO] Text found.")
        with open(output_path.replace('.jpg', '.txt'), 'w', encoding='utf-8') as f:
            f.write(text)


if __name__ == '__main__':
    main()
    
    
# TODO: Report
# TODO: Compare distance and the text here VS the text found by classical OCR software VS real text readed by human 