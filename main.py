from lib.agent import Agent
from lib.tools import Tools
from lib.ocr import OCR
from lib.img import Image

from tqdm import tqdm
import os

tools = Tools()
ocr = OCR()



# TODO: Modify with new Seorin Agent
def askLLMAgent(image_path, trans_lst, agent, N=1):

    location_path = "data_rag/BE_location_full.txt"
    
    agent = Agent(agent)
    text1 = agent.draft(image_path)    
    for _ in range(N):
        text2 = agent.refineLayout(text1, trans_lst)
        text3 = agent.checkNames(text2)
        text4 = agent.checkCities(text3, country = "Belgium", province = "Brabant wallon", municipality = "Nivelles", location_path = location_path, language = "French", lang="FR")
        
    agent.save_previous_documents(text4)
    
    return text4


def evaluate():
    img_lst = ["data/Archives_LLN_Nivelles_I_1921_REG 5193/example1.jpeg", "data/Archives_LLN_Nivelles_I_1921_REG 5193/example2.jpeg", "data/Archives_LLN_Nivelles_I_1921_REG 5193/example6.jpeg", "data/Archives_LLN_Nivelles_I_1921_REG 5193/example7.jpeg"]
    trans_lst = ["data/transcriptions/transcription_ex1.xlsx", "data/transcriptions/transcription_ex2.xlsx", "data/transcriptions/transcription_ex6.xlsx", "data/transcriptions/transcription_ex7.xlsx"]
    
    texts = {
        "Human": "",
        "GPT4o": "",
        "GPT4o cc": "",
        "GPT4": "",
        "GPT4 cc": "",
        "GPT4 turbo": "",
        "GPT4 turbo cc": "",
        "GPT3.5 turbo": "",
        "GPT3.5 turbo cc": "",
        "EasyOCR": "",
        "EasyOCR cc": "",
        "Pytesseract": "",
        "Pytesseract cc": "",
        "KerasOCR": "",
        "KerasOCR cc": "",
    }
    
    for i in tqdm(range(1)): #len(img_lst) # not all the images
        transcription = tools.xlsx_to_string(trans_lst[i])
        image_path = img_lst[i]
        
        img = Image(image_path)
        
        img.crop_image()
        img.color_image()
        output_path = image_path.replace('.jpeg', '_cc.jpeg')
        img.save(output_path)

        texts["Human"] += transcription + "\n"
        texts["GPT4o"] += askLLMAgent(image_path, trans_lst, "gpt-4o") + "\n"
        texts["GPT4o cc"] += askLLMAgent(output_path, trans_lst, "gpt-4o") + "\n"
        texts["GPT4"] += askLLMAgent(image_path, trans_lst, "gpt-4") + "\n"
        texts["GPT4 cc"] += askLLMAgent(output_path, trans_lst, "gpt-4") + "\n"
        texts["GPT4 turbo"] += askLLMAgent(image_path, trans_lst , "gpt-4-turbo") + "\n"
        texts["GPT4 turbo cc"] += askLLMAgent(output_path, trans_lst, "gpt-4-turbo") + "\n"
        texts["GPT3.5 turbo"] += askLLMAgent(image_path, trans_lst , "gpt-3.5-turbo") + "\n"
        texts["GPT3.5 turbo cc"] += askLLMAgent(output_path, trans_lst, "gpt-3.5-turbo") + "\n"
        texts["EasyOCR"] += ocr.easyOCR(image_path) + "\n"
        texts["EasyOCR cc"] += ocr.easyOCR(output_path) + "\n"
        texts["Pytesseract"] += ocr.pytesseractOCR(image_path) + "\n"
        texts["Pytesseract cc"] += ocr.pytesseractOCR(output_path) + "\n"
        texts["KerasOCR"] += ocr.kerasOCR(image_path) + "\n"
        texts["KerasOCR cc"] += ocr.kerasOCR(output_path) + "\n"

    tools.compare_texts(texts, "all")

def main():
    evaluate()

if __name__ == '__main__':
    main()
    
# TODO: compare methods in terms of running time