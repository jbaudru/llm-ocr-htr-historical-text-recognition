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
    
    def call(self, prompt, max_tokens=5000, base64_image=None, message=None, system=None):
        if("claude" in self.model):
            res = self.callAnthropic(prompt, max_tokens=max_tokens, base64_image=base64_image, message=message, system=system)
        else:
            res = self.callOpenAI(prompt, max_tokens=max_tokens, base64_image=base64_image, message=message)
        return res
    
    
    def callAnthropic(self, prompt, max_tokens=5000, base64_image=None, message=None, system=None):
        client = anthropic.Anthropic(api_key=self.anthropic_API_KEY)  
        try:
            if(message==None):
                if(base64_image):
                    response = client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        system = system,
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
                        temperature=0,
                    )
                else:
                    response = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    #system = system,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                }
                            ],
                        }
                    ],
                    temperature=0,
                )
            else:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    #system = system,
                    messages=message,
                    temperature=0,
                )
            
            return response.to_dict()["content"][0]["text"]
        except Exception as e:
            print(f"[ERROR] callAnthropic failed! {e}")
            return None
    
    
    def callOpenAI(self, prompt, max_tokens=5000, base64_image=None, message=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_API_KEY}"
        } 
        model_vision = "gpt-4o"
        if(base64_image):  
            if(message==None):
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
                    "temperature": 0
                }
            else:
                payload = {
                    "model": model_vision, # only gpt-4o can handle images
                    "messages": message,
                    "max_tokens": max_tokens,
                    "temperature": 0
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
        except Exception as e:
            print(f"[ERROR] callOpenAI failed! {e}")
            print(response.json()["error"]["message"])
        #return response
    
    #===========================================================================
    # AGENT FUNCTIONS
    #===========================================================================
    def draft(self, image_path, by_line=False):
        
        if("claude" in self.model):
            if(not by_line):
                resized_image = self.resize_image(image_path)
                base64_image = base64.b64encode(resized_image).decode('utf-8')
            else:
                base64_image = self.encode_image(image_path)
            
        else:
            base64_image = self.encode_image(image_path)
        
        system="You are a helpful assistant who can read old handwriting with a background in history, and you are going to recreate a scanned déclaration de succession from Belgium in a txt format."
            
        prompt = """
            Recognize the text from the image:
            ```plaintext
        """
            
        prompt = f"""
                From the example, you learned the handwriting of this Belgian record. You learned which alphabet and which number is written in which way.
                With this knowledge, now consider the following image to recreate:
                
                First, you read a two-level header in the table, which you recognize the same as the example as follows in the form of ("first level", "second level"):
                ```
                    [("N' d'ordre", " "),
                    ("Date du dépot des déclarations", " "),
                    ("Désignation des personnes décédées ou absentes.:", "Nom."),
                    ("Désignation des personnes décédées ou absentes.:", "Prénoms"),
                    ("Désignation des personnes décédées ou absentes.:", "Domiciles"), 
                    ("Date du décès ou du judgement d'envoi en possession, en cas d'absence.", " "),
                    ("Noms, Prénoms et demeures des parties déclarantes.", " "),
                    ("Droits de succession en ligne collatérale et de mutation en ligne directe.", "Actif. (2)"),
                    ("Droits de succession en ligne collatérale et de mutation en ligne directe.", "Passif. (2)"),
                    ("Droits de succession en ligne collatérale et de mutation en ligne directe.", "Restant NET. (2)"),
                    ("Droit de mutation par déces", "Valeur des immeubles. (2)"), 
                    ("Numéros des déclarations", "Primitives."),
                    ("Numéros des déclarations", "Supplémentaires."), 
                    ("Date", "de l'expiration du délai de rectification."),
                    ("Date", "de l'exigibilité des droits."),
                    ("Numéros de la consignation des droits au sommier n' 28", " "),
                    ("Recette des droits et amendes.", "Date"),
                    ("Recette des droits et amendes.", "N^03"),
                    ("Cautionnements. ", "Numéros de la consignation au sommier n'30"),
                    ("Observations (les déclarations qui figurent à l'état n'413 doivent être émargées en conséquence, dans la présnete colonne.)", " ")] 
                ```

                Context: 
                - It's written in French language and the names of the people are domiciles are Belgian.
                - Each row contains information about a dead person for the 20 variables above. Some rows contain information about the service date of the dead person written in the previous row. Such rows begin with texts like "Arrêté le \d{2} \w+ \d{4}( \w+)? servais" under "Nom." variable. 
                - When you see "Arrêté le \d{2} \w+ \d{4}( \w+)? servais", the subsequent row will be the next serviced day.
                - N' d'ordre will also follow an order. 
                - The family name in this column "Noms, Prénoms et demeures des parties déclarantes." may be the same as the family name in "Nom." column.
                
                Task: 
                Please recreate the table by filling in all the information in the record. Pay attention to reading each word and number correctly. 
                    ```plaintext
        """

        return self.call(prompt, max_tokens=3000, base64_image=base64_image, system=system)


    def callPostProcessing(self, context):
        base64_image = None
        prompt = "This is an output from you. Clean it such that we only have the table content without any comment from you:"
        prompt += context
        
        return self.call(prompt, max_tokens=3000, base64_image=base64_image)


    def exampleShot(self, image_path, NbExamples=1):
        # example
        example_xlsx = "data/transcriptions/transcription_ex" + str(2) + ".xlsx"
        example_text_1 = tools.xlsx_to_string(example_xlsx)
        example_image_1 = "data/Archives_LLN_Nivelles_I_1921_REG 5193/example2.jpeg"
        
        if(NbExamples==2):
            # example
            example_xlsx2 = "data/transcriptions/transcription_ex" + str(3) + ".xlsx"
            example_text_2 = tools.xlsx_to_string(example_xlsx2)
            example_image_2 = "data/Archives_LLN_Nivelles_I_1921_REG 5193/example3.jpeg"
        
        if("claude" in self.model):
            resized_image = self.resize_image(image_path)
            base64_image = base64.b64encode(resized_image).decode('utf-8')
            image_1 = self.resize_image(example_image_1)
            image_1 = base64.b64encode(image_1).decode('utf-8')
            if(NbExamples==2):
                image_2 = self.resize_image(example_image_2)
                image_2 = base64.b64encode(image_2).decode('utf-8')
        else:
            base64_image = self.encode_image(image_path)
            image_1 = self.encode_image(example_image_1)
            if(NbExamples==2):
                image_2 = self.encode_image(example_image_2)
        
        if("claude" in self.model):
            if(NbExamples==1):
                message = [
                    {
                        "role": "user",
                        "content": [ 
                        {"type": "image", 
                            "source": {
                                "type": "base64", 
                                "media_type": "image/jpeg", 
                                "data": image_1}
                        },
                        {
                            "type": "text",
                            "text": example_text_1,
                        },
                        {
                            "type": "text",
                            "text": f"""
                            The ```plaintext block is the example transcription of the example image you saw:

                            Transcription:
                            ```plaintext
                            {example_text_1}
                            ```
                            Compare what you read initially and the solution key in ```plaintext block. Recreate the content of the table in this image. Only that, no other information from you.

                            """
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        }
                        ]
                    }
                ]
            
            else:
                message = [
                    {
                        "role": "user",
                        "content": [ 
    
                        {"type": "image", 
                            "source": {
                                "type": "base64", 
                                "media_type": "image/jpeg", 
                                "data": image_1}},
                        {
                            "type": "text",
                            "text": example_text_1,
                        },
                        {"type": "image", 
                            "source": {
                                "type": "base64", 
                                "media_type": "image/jpeg", 
                                "data": image_2}},
                        {
                            "type": "text",
                            "text": example_text_2,
                        },
                        {
                            "type": "text",
                            "text": f"""
                            The ```plaintext block is the example transcription of the example image you saw:

                            Transcription:
                            ```plaintext
                            {example_text_1}
                            ```
                            Compare what you read initially and the solution key in ```plaintext block. Recreate the content of the table in this image. Only that, no other information from you.

                            """
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        }
                        ]
                    }
                ]
            
            system_prompt =  "You are a helpful assistant who can read old handwriting with a background in history, and you are going to recreate a scanned déclaration de succession from Belgium in a txt format."
            
        else:
            if(NbExamples==1):
                message = [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant who can read old handwriting with a background in history, and you are going to recreate a scanned déclaration de succession from Belgium in a txt format."
                    },
                    {
                        "role": "user",
                        "content": [ 
                        {
                            "type": "text",
                            "text": f"""
                            The ```plaintext block is the example transcription of the example image you saw:

                            Transcription:
                            ```plaintext
                            {example_text_1}
                            ```
                            Compare what you read initially and the solution key in ```plaintext block. Recreate the content of the table in this image. Only that, no other information from you.
                            
                            Even if it is hard to read the texts from the image, return as much as you can. You must read something. Do not return an apologetic message.
                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{image_1}"
                            }
                        },
                        ]
                    }
                ]
            
            else:
                message = [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant who can read old handwriting with a background in history, and you are going to recreate a scanned déclaration de succession from Belgium in a txt format."
                    },
                    {
                        "role": "user",
                        "content": [ 
                        {
                            "type": "text",
                            "text": f"""
                            The ```plaintext block is the example transcription of the example image you saw:

                            Transcription:
                            ```plaintext
                            {example_text_1}
                            ```
                            Compare what you read initially and the solution key in ```plaintext block. Recreate the content of the table in this image. Only that, no other information from you.
                            
                            Even if it is hard to read the texts from the image, return as much as you can. You must read something. Do not return an apologetic message.
                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{image_1}"
                            }
                        },
                        {
                            "type": "text",
                            "text": example_text_2
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{image_2}"
                            }
                        },
                        
                        ]
                    }
                ]
            # Force OpenAI
            # Even if it is hard to read the texts from the image, return as much as you can. You must read something. Do not return an apologetic message.

            system_prompt = None
        return self.call(prompt="", max_tokens=3000, base64_image=base64_image, message=message, system=system_prompt)
        
 
    def refineLayout(self, content): 
        
        prompt = f"""
            Your first draft:
            ```plaintext
            {content}
            ```

            Errors: 
            Your first transcription you made in ```plaintext block contains some errors.
            
            Task:
            Refine your first trasncription in ```plaintext block. 
            Make sure to read the names of the people and the location as well as the dates and the numbers correctly.
            Transcribe as you see in the image.
            ```plaintext
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
        