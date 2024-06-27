from lib.agent import Agent
from lib.tools import Tools
from lib.ocr import OCR
from lib.img import Image

agent = Agent()
tools = Tools()
ocr = OCR()


def askLLMAgent(image_path, N=1):
    text = agent.draft(image_path)
    print(text)
    for _ in range(N):
        text2 = agent.checkNames(text)
        text3 = agent.checkCities(text2)
        text4 = agent.checkMath(text3)
        text = agent.checkMath(text4)
        #text = agent.verifyContext(text5)
    agent.save_previous_documents(text)
    print(text)
    return text

def evaluate():
    img_lst = ["data/Archives_LLN_Nivelles_I_1921_REG 5193/example1.jpeg", "data/Archives_LLN_Nivelles_I_1921_REG 5193/example2.jpeg"]
    trans_lst = ["data/transcriptions/transcription_ex1.xlsx", "data/transcriptions/transcription_ex2.xlsx"]
    for i in range(len(img_lst)):
        transcription = tools.xlsx_to_string(trans_lst[i])
        image_path = img_lst[i]
        
        img = Image(image_path)
        
        img.crop_image()
        img.color_image()
        output_path = image_path.replace('.jpeg', '_cc.jpeg')
        img.save(output_path)
        
        texts = {
            "Human" : transcription,
            "LLM" : askLLMAgent(image_path),
            "LLM cc": askLLMAgent(output_path),
            "EasyOCR": ocr.easyOCR(image_path),
            "EasyOCR cc": ocr.easyOCR(output_path),
            "Pytesseract": ocr.pytesseractOCR(image_path),
            "Pytesseract cc": ocr.pytesseractOCR(output_path),
            "KerasOCR": ocr.kerasOCR(image_path),
            "KerasOCR cc": ocr.kerasOCR(output_path),
        }
        tools.compare_texts(texts, image_path)

def main():
    evaluate()

if __name__ == '__main__':
    main()
    
# TODO: compare methods in terms of running time