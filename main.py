from lib.agent import Agent
from lib.tools import Tools
from lib.ocr import OCR
from lib.img import Image

from tqdm import tqdm
import os

tools = Tools()
ocr = OCR()


# Iterative compiraison
def askLLMAgentFeedback(image_path, transcription, trans_lst, agent, N=10):
    agent = Agent(agent)
    i = 0; cer = 10; best_cer = 10
    text1 = ""; historic = ""
    
    while(i < N and cer > 0.2):
        text1 = agent.draft(image_path, historic) #agent.refineLayout(text1, image_path, trans_lst)
        
        cer = tools.CER(text1, transcription)
        if(cer < best_cer): 
            best_cer = cer
            
        agent.save_text(text1, image_path, "iter" + str(i) + "_")
            
        feedback = "Feedback:\n The CER (Character Error Rate) score for your previous output (below) was: " + str(cer) + "and your best score was:" + str(best_cer) + ". Improve your current score. Hre is your previous output:\n"
        historic += feedback + text1 
        print("CER (iter", str(i) ,"): ", cer)
        
        """
        text1 = agent.checkNames(text1)
        cer = tools.CER(text1, transcription)
        text1 += "Feedback: Your CER score is " + str(cer) + "try to improve that score" + "\n"
        print("CER: ", cer)
        text1 = agent.checkCities(text1, country = "Belgium", province = "Brabant wallon", municipality = "Nivelles", location_path = "data_rag/BE_location_full.txt", language = "French", lang="FR")
        cer = tools.CER(text1, transcription)
        text1 += "Feedback: Your CER score is " + str(cer) + "try to improve that score" + "\n"
        """
    
        i += 1


# Few-shot compiraison
def askLLMAgent(image_path, trans_lst, agent, N=1):
    #try:
        location_path = "data_rag/BE_location_full.txt"
        
        agent = Agent(agent)
        print("Agent: "+ agent.model)
        print(" - Drafting...")
        text1 = agent.draft(image_path)    
        for _ in range(N):
            print(" - Refining...")
            text2 = agent.refineLayout(text1, image_path, trans_lst)
            print(" - Checking name...")
            text3 = agent.checkNames(text2)
            print(" - Checking city...")
            text4 = agent.checkCities(text3, country = "Belgium", province = "Brabant wallon", municipality = "Nivelles", location_path = location_path, language = "French", lang="FR")
            
        #agent.save_previous_documents(text4)
        agent.save_text(text4, image_path)
        return text4
    #except:
    #    print("[ERROR] askLLMAgent failed!")
    #    return ""

# One shot compiraison
def askLLMAgentOneShot(image_path, trans_lst, agent):
    agent = Agent(agent)
    text1 = agent.draft(image_path)
    return text1



def append_result(texts, key, result):
    if result is not None:
        texts[key].append(result + "\n")
    else:
        texts[key].append("\n")
        print(f"Warning: {key} returned None for image_path")

def evaluate():
    # few-shot
    texts = {
        "GT": [],
        "gpt-4o": [],
        "gpt-4o-mini": [],
        "gpt-4": [],
        "gpt-4-turbo": [],
        "gpt-3.5-turbo-0125": [],
        "claude-3-5-sonnet-20240620": [],
        "EasyOCR": [],
        "Pytesseract": [],
        "KerasOCR": [],
    }
    
    #one-shot
    texts = {
        "GT": [],
        "gpt-4o": [],
        "claude-3-5-sonnet-20240620": [],
        "EasyOCR": [],
        "Pytesseract": [],
        "KerasOCR": [],
    }
    
    trans_lst = []
    img_lst = []
    for i in tqdm(range(0, 7), ascii=' >='): #8 max
        i += 1
        trans = "data/transcriptions/transcription_ex" + str(i) + ".xlsx"
        trans_lst.append(tools.xlsx_to_string(trans))
        
        image_path = "data/Archives_LLN_Nivelles_I_1921_REG 5193/example" + str(i) + ".jpeg"
        img_lst.append(image_path)
    
    for i in tqdm(range(len(img_lst)), ascii=' >='):
        transcription = trans_lst[i]
        image_path = img_lst[i]

        texts["GT"].append(transcription + "\n")

        #models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo-0125", "claude-3-5-sonnet-20240620"]
        models = ["gpt-4o", "claude-3-5-sonnet-20240620"]
        for model in models:
            result = askLLMAgentOneShot(image_path, trans_lst, model)
            append_result(texts, model, result)

        ocr_methods = {
            "EasyOCR": ocr.easyOCR,
            "Pytesseract": ocr.pytesseractOCR,
            "KerasOCR": ocr.kerasOCR
        }
        for key, method in ocr_methods.items():
            result = method(image_path)
            append_result(texts, key, result)

    tools.compare_texts_violin_plot(texts, "one-shot_simple-prompt")



def testCERFeedback():
    trans_lst = []
    for i in tqdm(range(1, 8), ascii=' >='): #len(img_lst) # not all the images (max7)
        trans = "data/transcriptions/transcription_ex" + str(i) + ".xlsx"
        trans_lst.append(tools.xlsx_to_string(trans))
    askLLMAgentFeedback("data/Archives_LLN_Nivelles_I_1921_REG 5193/example1.jpeg", trans_lst, "data/transcriptions/transcription_ex1.xlsx", "gpt-4o")

def testAnthropic():
    image_path = "data/Archives_LLN_Nivelles_I_1921_REG 5193/example1.jpeg"
    trans_lst = None
    res = askLLMAgentOneShot(image_path, trans_lst, "claude")
    print(res)

def main():
    evaluate()
    #testCERFeedback()
    #testAnthropic()

if __name__ == '__main__':
    main()
    
# TODO: compare methods in terms of running time