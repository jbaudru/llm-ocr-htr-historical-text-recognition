import base64
import requests
import hashlib
import os
from dotenv import load_dotenv

class Agent:
    def __init__(self) -> None:
        load_dotenv()  # Load environment variables from .env file
        self.openai_API_KEY = os.getenv("OPENAI_API_KEY")
        #self.openai_API_KEY = "sk-proj-26nXuqhTwwYPeP1PJleOT3BlbkFJgDKsQLeG7EeHUvh6sm2A"
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def save_previous_documents(self, content):
        directory = "data_rag/previous_doc"
        filename = str(hash(content)) + ".txt"
        filepath = os.path.join(directory, filename)

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(filepath, "w") as f:
            f.write(content)
    
    def load_previous_documents(self, content):
        # for each file in previous_doc, load the content, select the most common sentence
        pass
    
    def load_names(self):
        names = ""
        with open("data_rag/names.txt", "r") as f:
            names = f.read()
        return names

    def load_cities(self):
        cities = ""
        with open("data_rag/nivelles.txt", "r") as f:
            cities = f.read()
        return cities
    
    def call(self, prompt, base64_image):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_API_KEY}"
        } 
        if(base64_image):  
            payload = {
                "model": "gpt-4o",
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
                "max_tokens": 2000,
                "temperature": 0
            }
        else:
            payload = {
                "model": "gpt-4o",
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
                "max_tokens": 2000,
                "temperature": 0
            }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        # return response.json()["choices"][0]["message"]["content"]
        return response
    
    def draft(self, image_path, output_format = "json"):
        base64_image = self.encode_image(image_path)
        prompt = f"You are a helpful assistant, recreate the table from this handwritten document into a {output_format} file, this table contain columns and subcolums. No PROFESSION column. Copy the texts as they are, do not add any other sentences from you:"
        return self.call(prompt, base64_image)
    
    def refineLayout(self, draft):
        
    prompt = f"""
        Your first draft is 
        '''
        {draft}
        '''

        This document is a Déclaration de succession of Belgium and expressed in a nested json, it should have the following structure:
            \{
                "N' d'ordre": \{
                    "Unnamed: 0_level_1": 
                \},
                "Date du dépot des déclarations": \{
                    "Unnamed: 1_level_1": 
                \},
                "Désignation des personnes décédées ou absentes.": \{
                    "Nom.":  ,
                    "Prénoms":  ,
                    "Domiciles": ,
                    "Domiciles.1": "
                \},
                "Date du décès ous du judgement d'envoi en possession, en cas d'absence.": \{
                    "Unnamed: 6_level_1": 
                \},
                "Noms, Prénoms et demeures des parties déclarantes.": \{
                    "Unnamed: 7_level_1": 
                \},
                "Droits de succession en ligne collatérale et de mutation en ligne directe.": \{
                    "Actif. (2)": ,
                    "Passif. (2)": ,
                    "Restant NET. (2)": 
                \},
                "Droit de mutation par déces": \{
                    "Valeur des immeubles. (2)": 
                \},
                "Numéros des déclarations": \{
                    "Primitives.": ,
                    "Supplémentaires.": 
                \},
                "Date": \{
                    "de l'expiration du délai de rectification.": ,
                    "de l'exigibilité des droits.": 
                \},
                "Numéros de la consignation des droits au sommier n' 28": \{
                    "Unnamed: 16_level_1": 
                \},
                "Recette des droits et amendes.": \{
                    "Date": ,
                    "N^03": 
                \},
                "Cautionnements. ": \{
                    "Numéros de la consignation au sommier n'30": 
                \},
                "Observations": \{
                    "(les déclarations qui figurent à l'état n'413 doivent être émargées en conséquence, dans la présnete colonne.)": 
                \}
            \}

        Please refine your first draft based on this structure.

        """
    return self.call(prompt)
    
    def checkNames(self, content):
        prompt = "Verify this table. It should containt Belgian family names and first name, there is high probability that the family names appear mutliple times in a same row. I want a table in .txt format as output, just the table no other sentence from you:"
        prompt += content
        prompt += self.load_names()
        return self.call(prompt, None)
    
    def checkCities(self, content):
        prompt = "Verify this table. It should containt Belgian cities and municipality, there is high probability that the cities appear mutliple times in a same column. I want a table in .txt format as output, just the table no other sentence from you:"
        prompt += content
        prompt += self.load_cities()
        return self.call(prompt, None)
    
    def checkMath(self, content):
        prompt = "Verify this table, in the column 'Droit de succession', the values in the subcolumns 'Rest' = 'Actif' - 'Passif'. I want a table in .txt format as output, juste the table no other sentence from you:"
        prompt += content
        return self.call(prompt, None)
    
    def verifyContext(self, content):
        prompt = "Verify this table. It shoud containt sentences from the following list. I want a table in .txt format as output, juste the table no other sentence from you:"
        prompt += content
        # TODO: add most common sentence in the previous documents
        return self.call(prompt, None)
        