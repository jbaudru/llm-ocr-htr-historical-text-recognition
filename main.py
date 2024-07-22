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
    try:
        location_path = "data_rag/BE_location_full.txt"
        
        agent = Agent(agent)
        print("Agent: "+ agent.model)
        print(" - Drafting...")
        text1 = agent.draft(image_path)    
        for _ in range(N):
            print(" - Refining...")
            text2 = agent.refineLayout(text1, trans_lst)
            print(" - Checking name...")
            text3 = agent.checkNames(text2)
            print(" - Checking city...")
            text4 = agent.checkCities(text3, country = "Belgium", province = "Brabant wallon", municipality = "Nivelles", location_path = location_path, language = "French", lang="FR")
            
        #agent.save_previous_documents(text4)
        agent.save_text(text4, image_path)
        return text4
    except:
        print("[ERROR] askLLMAgent failed!")
        return ""


def evaluate():
    texts = {
        "Human": "",
        "GPT4o": "",
        "GPT4o mini": "",
        "GPT4": "",
        "GPT4 turbo": "",
        "GPT3.5 turbo": "",
        "EasyOCR": "",
        "Pytesseract": "",
        "KerasOCR": "",
    }
    
    texts_cc = {
        "Human": "",
        "GPT4o cc": "",
        "GPT4o mini cc": "",
        "GPT4 cc": "",
        "GPT4 turbo cc": "",
        "GPT3.5 turbo cc": "",
        "EasyOCR cc": "",
        "Pytesseract cc": "",
        "KerasOCR cc": "",
    }
    
    
    trans_lst = []
    img_lst = []
    for i in tqdm(range(1, 8), ascii=' >='): #len(img_lst) # not all the images (max7)
        trans = "data/transcriptions/transcription_ex" + str(i) + ".xlsx"
        trans_lst.append(tools.xlsx_to_string(trans))
        
        image_path = "data/Archives_LLN_Nivelles_I_1921_REG 5193/example" + str(i) + ".jpeg"
        img_lst.append(image_path)
    
    for i in tqdm(range(len(img_lst)), ascii=' >='):
        transcription = trans_lst[i]
        
        image_path = img_lst[i]
        #img = Image(image_path)
        #img.crop_image()
        #img.color_image()
        #output_path = image_path.replace('.jpeg', '_cc.jpeg')
        #img.save(output_path)

        texts["Human"] += transcription + "\n"
        texts["GPT4o"] += askLLMAgent(image_path, trans_lst, "gpt-4o") + "\n"
        texts["GPT4o mini"] += askLLMAgent(image_path, trans_lst, "gpt-4o-mini") + "\n"
        #texts["GPT4o cc"] += askLLMAgent(output_path, trans_lst, "gpt-4o") + "\n"
        texts["GPT4"] += askLLMAgent(image_path, trans_lst, "gpt-4") + "\n"
        #texts["GPT4 cc"] += askLLMAgent(output_path, trans_lst, "gpt-4") + "\n"
        texts["GPT4 turbo"] += askLLMAgent(image_path, trans_lst , "gpt-4-turbo") + "\n"
        #texts["GPT4 turbo cc"] += askLLMAgent(output_path, trans_lst, "gpt-4-turbo") + "\n"
        texts["GPT3.5 turbo"] += askLLMAgent(image_path, trans_lst , "gpt-3.5-turbo") + "\n"
        #texts["GPT3.5 turbo cc"] += askLLMAgent(output_path, trans_lst, "gpt-3.5-turbo") + "\n"
        texts["EasyOCR"] += ocr.easyOCR(image_path) + "\n"
        #texts["EasyOCR cc"] += ocr.easyOCR(output_path) + "\n"
        texts["Pytesseract"] += ocr.pytesseractOCR(image_path) + "\n"
        #texts["Pytesseract cc"] += ocr.pytesseractOCR(output_path) + "\n"
        texts["KerasOCR"] += ocr.kerasOCR(image_path) + "\n"
        #texts["KerasOCR cc"] += ocr.kerasOCR(output_path) + "\n"

    tools.compare_texts(texts, "all")

def main():
    evaluate()

if __name__ == '__main__':
    main()
    
# TODO: compare methods in terms of running time