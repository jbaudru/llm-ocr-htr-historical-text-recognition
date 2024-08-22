from lib.agent import Agent
from lib.tools import Tools
from lib.ocr import OCR
from lib.img import Image

from tqdm import tqdm

import os
import mlflow
import time

current_dir = os.getcwd()
mlflow.set_tracking_uri(f"file:///{current_dir}/mlruns")
mlflow.set_experiment("LLM-HTR")

tools = Tools()
ocr = OCR()


# Iterative compiraison
def askLLMAgentFeedback(image_path, transcription, trans_lst, agent, N=10):
    agent = Agent(agent)
    i = 0; cer = 10; best_cer = 10
    text1 = ""; historic = ""
    
    while(i < N and cer > 0.2):
        text1 = agent.draft(image_path, historic) #agent.refineLayout(text1, image_path, trans_lst)
        
        cer = tools.CER(agent, text1, transcription)
        if(cer < best_cer): 
            best_cer = cer
            
        agent.save_text(text1, image_path, "iter" + str(i) + "_")
            
        feedback = "Feedback:\n The CER (Character Error Rate) score for your previous output (below) was: " + str(cer) + "and your best score was:" + str(best_cer) + ". Improve your current score. Hre is your previous output:\n"
        historic += feedback + text1 
        print("CER (iter", str(i) ,"): ", cer)
            
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
            #print(" - Checking name...")
            #text3 = agent.checkNames(text2)
            #print(" - Checking city...")
            #text4 = agent.checkCities(text3, country = "Belgium", province = "Brabant wallon", municipality = "Nivelles", location_path = location_path, language = "French", lang="FR")
            
        #agent.save_previous_documents(text4)
        agent.save_text(text2, image_path)
        return text2
    #except:
    #    print("[ERROR] askLLMAgent failed!")
    #    return ""

# zero shot compiraison
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
    experiment_name = "zero-shot_simple-prompt"
    
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
    
    #zero-shot
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
    for i in tqdm(range(0, 20), ascii=' >='): #20 max
        i += 1
        trans = "data/transcriptions/transcription_ex" + str(i) + ".xlsx"
        trans_lst.append(tools.xlsx_to_string(trans))
        
        image_path = "data/Archives_LLN_Nivelles_I_1921_REG 5193/example" + str(i) + ".jpeg"
        img_lst.append(image_path)
    
    # Create experiment folder
    experiment_folder = os.path.join("mlflow_experiments", experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)
    
    for i in tqdm(range(len(img_lst)), ascii=' >='):
        transcription = trans_lst[i]
        image_path = img_lst[i]

        texts["GT"].append(transcription + "\n")

        #models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo-0125", "claude-3-5-sonnet-20240620"]
        models = ["gpt-4o", "claude-3-5-sonnet-20240620"]
        for model in models:
            method_folder = os.path.join(experiment_folder, model)
            os.makedirs(method_folder, exist_ok=True)
            
            with mlflow.start_run():
                mlflow.log_param("method", model)
                result = askLLMAgentOneShot(image_path, trans_lst, model)
                mlflow.log_param("method", model)
                mlflow.log_metric("cer", (tools.CER(model, result, transcription)))
                append_result(texts, model, result)
                
                # Save result to file
                result_file = os.path.join(method_folder, f"transcription_{i+1}.txt")
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(result)

                # add waiting time to avoid overloading the server
                time.sleep(5)
                
        ocr_methods = {
            "EasyOCR": ocr.easyOCR,
            "Pytesseract": ocr.pytesseractOCR,
            "KerasOCR": ocr.kerasOCR
        }
        for key, method in ocr_methods.items():
            method_folder = os.path.join(experiment_folder, key)
            os.makedirs(method_folder, exist_ok=True)
            
            with mlflow.start_run():
                mlflow.log_param("method", key)
                result = method(image_path)
                mlflow.log_param("method", key)
                mlflow.log_metric("cer", (tools.CER(key, result, transcription)))
                append_result(texts, key, result)
                
                # Save result to file
                result_file = os.path.join(method_folder, f"transcription_{i+1}.txt")
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(result)

    tools.compare_texts_violin_plot(texts, experiment_name)
    
    
def main():
    evaluate()


if __name__ == '__main__':
    main()
    