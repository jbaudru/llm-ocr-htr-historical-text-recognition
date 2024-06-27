import base64
import requests
import hashlib
import os
class Agent:
    def __init__(self) -> None:
        self.openai_API_KEY = "sk-proj-PwHJjpWHrxzyPJxQ8W0tT3BlbkFJk7rrTpTkpYZUI5L57Gf9"
        pass
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def save_previous_documents(self, content):
        directory = "previous_doc"
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
                "max_tokens": 1000,
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
                "max_tokens": 1000,
                "temperature": 0
            }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]
    
    def draft(self, image_path):
        base64_image = self.encode_image(image_path)
        prompt = "You are a helpful assistant, recreate the table from this handwritten document, this table contain columns and subcolums. No PROFESSION column. I want a table in .txt format as output, juste the table no other sentence from you:"
        return self.call(prompt, base64_image)
    
    def checkNames(self, content):
        prompt = "Verify this table. It should containt Belgian family names and first name, there is high probability that the family names appear mutliple times in a same row. I want a table in .txt format as output, juste the table no other sentence from you:"
        prompt += content
        prompt += self.load_names()
        return self.call(prompt, None)
    
    def checkCities(self, content):
        prompt = "Verify this table. It should containt Belgian cities and municipality, there is high probability that the cities appear mutliple times in a same column. I want a table in .txt format as output, juste the table no other sentence from you:"
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
        