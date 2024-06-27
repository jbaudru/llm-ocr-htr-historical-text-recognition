import base64
import requests

class Agent:
    def __init__(self) -> None:
        self.openai_API_KEY = "sk-proj-26nXuqhTwwYPeP1PJleOT3BlbkFJgDKsQLeG7EeHUvh6sm2A"
        pass
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def call(self, image_path):
        base64_image = self.encode_image(image_path)
        #prompt = "Give me the text in the image (a table), only the text that you are able to read, correct the text if there is missing information or typo, no additional information, no markdown element only '\n':"
        prompt = "Recognize the text in the image and correct it if necessary, the image contain a table and I want a table in .txt format as output, juste the table no other sentence from you:"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_API_KEY}"
        }    
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
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]