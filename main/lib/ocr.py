from PIL import Image
import cv2
import easyocr
import pytesseract
import keras_ocr
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Download: https://github.com/UB-Mannheim/tesseract/wiki

# On Asus VivoBook & Office PC
#pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
# On Home PC
pytesseract.pytesseract.tesseract_cmd = "C:/Users/julien/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"

class OCR:
    
    def easyOCR(self, image_path):
        reader = easyocr.Reader(['fr'])
        img = cv2.imread(image_path)
        results = reader.readtext(img)
        output = []
        for res in results:
            det, conf = res[1], res[2]
            output.append((det, round(conf, 2))) 
        text = ' '.join([i[0] for i in output])
        return text
    
    # TO test
    def trOCR(self, image_path):
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        image = Image.open(image_path)
        pixel_values = processor(image, return_tensors="pt").pixel_values
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pixel_values = pixel_values.to(device)

        generated_ids = model.generate(pixel_values, max_new_tokens=4000)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    

    def pytesseractOCR(self, image_path):
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text
        except:
            print("[ERROR] pytesseractOCR failed! (should be installed)")
            return ""

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