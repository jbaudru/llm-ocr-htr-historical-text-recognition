import base64
import requests
import hashlib
import random
import os
from PIL import Image
from io import BytesIO
import pandas as pd
from dotenv import load_dotenv
from lib.tools import Tools
import anthropic

tools = Tools()

class Agent:
    def __init__(self, model) -> None:
        load_dotenv()  # Load environment variables from .env file
        self.openai_API_KEY = os.getenv("OPENAI_API_KEY")
        self.anthropic_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.model = model
    
    #===========================================================================
    # UTILITIES FUNCTIONS
    #===========================================================================
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def resize_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                # Calculate new dimensions
                new_width = img.width // 3
                new_height = img.height // 3
                img = img.resize((new_width, new_height), Image.LANCZOS)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                return buffer.getvalue()
        except Exception as e:
            print(f"[ERROR] Resizing image: {e}")
            return None
    
    def save_previous_documents(self, content):
        directory = "data_rag/previous_doc"
        filename = str(hash(content)) + ".txt"
        filepath = os.path.join(directory, filename)

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(filepath, "w") as f:
            f.write(content)

    def save_text(self, text, image_path, suffix=""):
        directory = "results/Results_Prediction"
        filename = image_path.split("/")[-1].replace(".jpeg", ".txt")
        filepath = directory + "/" + "pred_" + suffix + filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
    
    def load_previous_documents(self, content):
        # for each file in previous_doc, load the content, select the most common sentence
        pass
    
    def load_names(self):
        names = ""
        #with open("data_rag/names.txt", "r") as f:
        with open("data_rag/names_short.txt", "r") as f:
            names = f.read()
        return names

    def load_cities(self):
        cities = ""
        with open("data_rag/nivelles.txt", "r") as f:
            cities = f.read()
        return cities
    
    #===========================================================================
    # LLM CALLS
    #===========================================================================
    
    def call(self, prompt, max_tokens=5000, base64_image=None):
        if(self.model == "claude"):
            res = self.callAnthropic(prompt, max_tokens=max_tokens, base64_image=base64_image)
        else:
            res = self.callOpenAI(prompt, max_tokens=max_tokens, base64_image=base64_image)
        return res
    
    
    def callAnthropic(self, prompt, max_tokens=5000, base64_image=None):
        model = "claude-3-5-sonnet-20240620"
        client = anthropic.Anthropic(api_key=self.anthropic_API_KEY)  
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    }
                ],
            )
            return response.to_dict()["content"][0]["text"]
        except Exception as e:
            print(f"[ERROR] callAnthropic failed! {e}")
            return None
    
    def callOpenAI(self, prompt, max_tokens=5000, base64_image=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_API_KEY}"
        } 
        model_vision = "gpt-4o"
        if(base64_image):  
            payload = {
                "model": model_vision, # only gpt-4o can handle images
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
                "max_tokens": max_tokens,
                "temperature": 0.25
            }
        else:
            payload = {
                "model": self.model,
                "messages": [
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                    ]
                }
                ],
                "max_tokens": max_tokens,
                "temperature": 0
            }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            return response.json()["choices"][0]["message"]["content"]
        except:
            print(response.json()["error"]["message"])
        #return response
    
    #===========================================================================
    # AGENT FUNCTIONS
    #===========================================================================
    def draft(self, image_path, feedback="", output_format = "txt"):
        if(self.model == "claude"):
            resized_image = self.resize_image(image_path)
            base64_image = base64.b64encode(resized_image).decode('utf-8')
        else:
            base64_image = self.encode_image(image_path)
        prompt = "Recreate the content of the table in this image. Only that, no other information from you."
        return self.call(prompt, max_tokens=3000, base64_image=base64_image)

    # TODO: Add the image ? 
    def refineLayout(self, content, image_path, transcription_lst): 
        transcriptions = ""
        
        # Take a random transcription as an example and avoid the same transcription
        img_number = image_path.split("/")[-1].replace(".jpeg", "")
        index = random.randint(0, len(transcription_lst)-1)
        while img_number == index:
            index = random.randint(0, len(transcription_lst)-1)
        transcriptions += transcription_lst[index] 
        
        prompt = f"""
            Your first draft:
                ```draft
                {content}
                ```
            
            Example:
                ```txt
                {transcriptions}
                ```  

            Errors: 
                Your first draft in the ```draft block contains some errors. 
            
            Context: 
                The content of your draft in ```draft block should follow the structure in the example in the ```txt block. 
                Information for 'Actif.', 'Passif.' and 'Restant Net.' should exist for each dead person.
            
            Task:
                Refine your first draft based on the example.
                When you see Arrêté le \d{2} \w+ \d{4}( \w+)? servais, add it to a new key key called 'Note'. 
                If there is no information of the deceased name but only 'Arrêté le \d{2} \w+ \d{4}( \w+)? servais', add it under an empty name.
                Make sure to read the names of the people and the location as well as the dates and the numbers correctly.
                Don't add any other information, just the table.
            
        """
        return self.call(prompt, max_tokens=3000)
    
    """
    def checkNames(self, content):
        prompt = "Verify this table. It should containt Belgian family names and first name, there is high probability that the family names appear mutliple times in a same row. I want a table in .txt format as output, just the table no other sentence from you:"
        prompt += content
        prompt += self.load_names()
        return self.call(prompt, None)
    """
    
    def checkNames(self, content):
        prompt = f"""
        Your first draft:
            ```draft
            {content}
            ```
        
        Name list:
            ```txt
            {self.load_names()}
            ```  
        
        Task:
            There are some transcription errors in 'Nom', 'Prénoms', and 'Noms, Prénoms et demeures des parties déclarantes' in your first draft in the ```draft block.
            Read these items from the image again such that the corresponding names exist in Belgium according to the name list in the ```txt block.
            When you see Arrêté le \d{2} \w+ \d{4}( \w+)? servais, add it to a new key key called 'Note'. 
            If there is no information of the deceased name but only 'Arrêté le \d{2} \w+ \d{4}( \w+)? servais', add it under an empty name.
            Make sure to read the names of the people and the location as well as the dates and the numbers correctly.
            Only update those items. 
            Don't add any other information, just the table.
        
        Tips:
            The family name in 'Nom' under 'Désignation des personnes décédées ou absentes.' may equal to the family name in 'Noms, Prénoms et demeures des parties déclarantes', which contains the family and first name of the declaring parties.
            But it is most likely that their first names (Prénoms) are different. 
            If the family names in 'Nom' and 'Noms, Prénoms et demeures des parties déclarantes' are the same, 'Prénoms' should be masculine.
            'Noms, Prénoms et demeures des parties déclarantes' may end with '& autre' or '& autres'.
    
        """
        return self.call(prompt, max_tokens=3000)
    
    
    """
    def checkCities(self, content):
        prompt = "Verify this table. It should containt Belgian cities and municipality, there is high probability that the cities appear mutliple times in a same column. I want a table in .txt format as output, just the table no other sentence from you:"
        prompt += content
        prompt += self.load_cities()
        return self.call(prompt, None)
    """
    
    def checkCities(self, content, country, province, municipality, location_path, language="French", lang="FR"):
        #txt = pd.read_csv(location_path, sep='\t')
        #province = txt[txt['Province'] == {province}].copy()
        province = self.load_cities()
        prompt = f"""
        
        Your draft:
            ```draft
            {content}
            ```
        
        Province data:
            ```txt
            {province}
            ```
        
        Task:
            There may be errors in the information you filled in 'Domiciles' in your first draft in the ```draft block. Refine it.
            To improve 'Domiciles' in your draft, consult Sector_{lang} in the province data in ```txt block. 
            When you see Arrêté le \d{2} \w+ \d{4}( \w+)? servais, add it to a new key key called 'Note'. 
            If there is no information of the deceased name but only 'Arrêté le \d{2} \w+ \d{4}( \w+)? servais', add it under an empty name.
            Make sure to read the names of the people and the location as well as the dates and the numbers correctly.
            Don't add any other information, just the table.
            
        Tips:
            The sector names in your draft and the province data may slightly differ.
            'Domiciles' in your draft should contain sector names in Sector_{lang}.

        
        """
        return self.call(prompt, max_tokens=3000)
    
    def checkMath(self, content):
        prompt = "Verify this table, in the column 'Droit de succession', the values in the subcolumns 'Rest' = 'Actif' - 'Passif'. I want a table in .txt format as output, juste the table no other sentence from you:"
        prompt += content
        return self.call(prompt, None)
    
    def verifyContext(self, content):
        prompt = "Verify this table. It shoud containt sentences from the following list. I want a table in .txt format as output, juste the table no other sentence from you:"
        prompt += content
        # TODO: add most common sentence in the previous documents
        return self.call(prompt, None)
        