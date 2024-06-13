from PIL import Image
import cv2
import easyocr
import pytesseract
import keras_ocr

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
# https://github.com/UB-Mannheim/tesseract/wiki

class OCR:
    
    def easyOCR(self, image_path):
        reader = easyocr.Reader(['en'])
        img = cv2.imread(image_path)
        results = reader.readtext(img)
        output = []
        for res in results:
            det, conf = res[1], res[2]
            output.append((det, round(conf, 2))) 
        text = ' '.join([i[0] for i in output])
        return text


    def pytesseractOCR(self, image_path):
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text


    def kerasOCR(self, image_path):
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