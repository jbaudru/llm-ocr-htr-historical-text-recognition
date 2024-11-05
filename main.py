from lib.agent import Agent
from lib.tools import Tools
from lib.ocr import OCR
from lib.img import Image

from tqdm import tqdm
import pandas as pd

import os
import mlflow
import time
import glob

current_dir = os.getcwd()
mlflow.set_tracking_uri(f"file:///{current_dir}/mlruns")
mlflow.set_experiment("LLM-HTR")

tools = Tools()
ocr = OCR()


def append_result(texts, key, result):
    if result is not None:
        texts[key].append(result + "\n")
    else:
        texts[key].append("\n")
        print(f"Warning: {key} returned None for image_path")



def evaluate():
    #experiment_name = "zero-shot_simple-prompt"
    #experiment_name = "zero-shot_complex-prompt"
    experiment_name = "one-example_prompt"
    #experiment_name = "two-example_prompt"
    #experiment_name = "refine_complex-prompt"
    
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
        "trOCR": []
    }
    
    #zero-shot
    texts = {
        "GT": [],
        "gpt-4o": [],
        "claude-3-5-sonnet-20240620": [],
    }
    
    all = []
    trans_lst = []
    img_lst = []
    
    for i in tqdm(range(0, 20), ascii=' >='): #20 max
        i += 1
        #trans = "data/transcriptions/transcription_ex" + str(i) + ".xlsx"
        #trans_txt = tools.xlsx_to_string(trans)
        # save transcription into a text file
        with open(f"data/transcriptions/transcription_ex{i}.txt", "r", encoding="utf-8") as f:
            trans_txt = f.read()
            #print(trans_txt)

        trans_lst.append(trans_txt)
        
        image_path = f"data/Archives_LLN_Nivelles_I_1921_REG 5193/example{i}.jpeg"
        img_lst.append(image_path)

    print(f"Number of image: {len(img_lst)}")
    print(f"Number of trans: {len(trans_lst)}")    
    
    # Create experiment folder
    experiment_folder = os.path.join("results/predictions/", experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)
    
    for i in tqdm(range(len(img_lst)), ascii=' >='):
        print("Processing image", img_lst[i])

        transcription = trans_lst[i]
        image_path = img_lst[i]

        texts["GT"].append(transcription + "\n")

        #models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo-0125", "claude-3-5-sonnet-20240620"]
        #models = ["gpt-4o", "claude-3-5-sonnet-20240620"]
        models = ["claude-3-5-sonnet-20240620"]
        for model in models:
            agent = Agent(model)
            method_folder = os.path.join(experiment_folder, model)
            os.makedirs(method_folder, exist_ok=True)

            
            #if(i not in [0,1,5,7,13]):
                
            # Zero-shot
            #result = agent.draft(image_path, by_line=False)
            #result = agent.callPostProcessing(res)
            
            # One-example / Two-example
            result = agent.exampleShot(image_path, NbExamples=2)
            while ("sorry" in result):
                result = agent.exampleShot(image_path, NbExamples=2)
            print(result)
            
            # Refine
            #result = agent.refineLayout(res)

            
            #mlflow.log_param("method", model)
            #mlflow.log_metric("cer", (tools.compute_distances(result, transcription)[-2]))
            #mlflow.log_metric("bleu", (tools.compute_distances(result, transcription)[-1]))
            #append_result(texts, model, result)
            
            #all.append({'file': i, 'model': model, 'res': result})
            
            # Save result to file
            result_file = os.path.join(method_folder, f"new_transcription{i}.txt")
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(result)

            # add waiting time to avoid overloading the server
            time.sleep(2.5)
        
        """
        ocr_methods = {
            "EasyOCR": ocr.easyOCR,
            "Pytesseract": ocr.pytesseractOCR,
            "KerasOCR": ocr.kerasOCR,
            "trOCR": ocr.trOCR
        }
        
        for key, method in ocr_methods.items():
            method_folder = os.path.join(experiment_folder, key)
            os.makedirs(method_folder, exist_ok=True)
            
            with mlflow.start_run():
                mlflow.log_param("method", key)
                result = method(image_path)
                
                mlflow.log_param("method", key)
                mlflow.log_metric("cer", (tools.compute_distances(result, transcription)[-2]))
                mlflow.log_metric("bleu", (tools.compute_distances(result, transcription)[-1]))
                append_result(texts, key, result)
                
                all.append({'file': i, 'model': key, 'res': result})
                
                # Save result to file
                result_file = os.path.join(method_folder, f"transcription{i}.txt")
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(result)

        all_df = pd.DataFrame(all)
        file_path = f"results/predictions/{experiment_name}/all.csv"

        if os.path.isfile(file_path):
            all_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            all_df.to_csv(file_path, mode='w', header=True, index=False)
        """

    #tools.compare_texts_violin_plot(texts, experiment_name)
    

    
def main():
    evaluate()
    #evaluate_line_by_line()


if __name__ == '__main__':
    main()
    