from openai import OpenAI
from anthropic import Anthropic
import time
import json
import os
import re
import pandas as pd
from dotenv import load_dotenv
import requests
import base64
import subprocess
from IPython.display import display, Image
from PIL import Image as PILImage

# OpenAI
load_dotenv() 
openai_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_API_KEY)
gpt_model = "gpt-4o-2024-08-06"

# Anthropic
anthropic_API_KEY = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = Anthropic(api_key=anthropic_API_KEY)
anthropic_model = "claude-3-5-sonnet-20240620"

#================================================================================================
# LLM Calls
#================================================================================================

## OpenAI Basic =================================================================================================

def callOpenAI(prompt, max_tokens=800, base64_image=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_API_KEY}"
    } 
    model_vision = gpt_model
    payload = {
        "model": model_vision, 
        "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant who can read old handwriting with a background in history, and you are going to recreate a scanned déclaration de succession from Belgium in a txt format."
            
        },
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
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except:
        print(response.json()["error"]["message"])

## Anthropic Basic =================================================================================================
        
def callAnthropic(prompt, max_tokens=5000, base64_image=None):
    response = anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=max_tokens,
        system = "You are a helpful assistant who can read old handwriting with a background in history, and you are going to recreate a scanned déclaration de succession from Belgium in a txt format.",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", 
                        "source": {
                            "type": "base64", 
                            "media_type": "image/jpeg", 
                            "data": base64_image}},
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
        temperature=0,
    )
    return response.to_dict()["content"][0]["text"]

## Post-processing =================================================================================================
def callPostProcessing(max_tokens=800, prompt_parameter = None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_API_KEY}"
    } 
    payload = {
        "model": gpt_model,
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": f"""This is an output from you. Clean it such that we have no separators and no comment from you: {prompt_parameter}
                """
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

# use this when OpenAI credits are exhausted
def callPostProcessing_anthropic(max_tokens=5000, prompt_parameter = None):
    response = anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=max_tokens,
        system = "You are a helpful assistant who can read old handwriting with a background in history, and you are going to recreate a scanned déclaration de succession from Belgium in a txt format.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""This is an output from you. Clean it such that we have no separators and no comment from you: {prompt_parameter}
                """
                    }
                ],
            }
        ],
        temperature=0,
    )
    return response.to_dict()["content"][0]["text"]


## Few-shots =================================================================================================

def callOpenAI_example(prompt, NExample=1, base64_image=None, max_tokens=5000):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_API_KEY}"
    } 
    model_vision = gpt_model

    if NExample == 1:
        payload = {
            "model": model_vision, 
            "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant who can read old handwriting with a background in history, and you are going to recreate a scanned déclaration de succession from Belgium in a txt format."
                
            },
            {
                "role": "user",
                "content": [
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{example1}"
                    }
                },
                {
                    "type": "text",
                    "text": example1_text
                },
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
    if NExample == 2:
               payload = {
            "model": model_vision, 
            "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant who can read old handwriting with a background in history, and you are going to recreate a scanned déclaration de succession from Belgium in a txt format."
                
            },
            {
                "role": "user",
                "content": [
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{example1}"
                    }
                },
                {
                    "type": "text",
                    "text": example1_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{example2}"
                    }
                },
                {
                    "type": "text",
                    "text": example2_text
                },
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
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except:
        print(response.json()["error"]["message"])
        

def callAnthropic_example(prompt, NExample=1, base64_image=None, max_tokens=5000):
    if NExample == 1:
        response = anthropic_client.messages.create(
            model=anthropic_model,
            max_tokens=max_tokens,
            system = "You are a helpful assistant who can read old handwriting with a background in history, and you are going to recreate a scanned déclaration de succession from Belgium in a txt format.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", 
                            "source": {
                                "type": "base64", 
                                "media_type": "image/jpeg", 
                                "data": example1}},
                        {
                            "type": "text",
                            "text": example1_text,
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {"type": "image", 
                            "source": {
                                "type": "base64", 
                                "media_type": "image/jpeg", 
                                "data": base64_image}}
                    ],
                }
            ],
            temperature=0,
        )
        
    if NExample == 2:
        response = anthropic_client.messages.create(
            model=anthropic_model,
            max_tokens=max_tokens,
            system = "You are a helpful assistant who can read old handwriting with a background in history, and you are going to recreate a scanned déclaration de succession from Belgium in a txt format.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", 
                            "source": {
                                "type": "base64", 
                                "media_type": "image/jpeg", 
                                "data": example1}},
                        {
                            "type": "text",
                            "text": example1_text,
                        },
                        {"type": "image", 
                            "source": {
                                "type": "base64", 
                                "media_type": "image/jpeg", 
                                "data": example2}},
                        {
                            "type": "text",
                            "text": example2_text,
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {"type": "image", 
                            "source": {
                                "type": "base64", 
                                "media_type": "image/jpeg", 
                                "data": base64_image}}
                    ],
                }
            ],
            temperature=0,
        )
    return response.to_dict()["content"][0]["text"]





#================================================================================================
# Prompts
#================================================================================================


## Zero-shot
prompt_simple = """
    Recognize the text from the image:
    ```plaintext
"""


prompt_complex = f"""
    Context:
        It's an old Belgian document. And you're getting one row of a table from it. It's written in French language and the names of the people are domiciles are Belgian.

    Structure:
        The table is structured with the two-level headers as follows:
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
                ("Droit de mutation par décès", "Valeur des immeubles. (2)"), 
                ("Numéros des déclarations", "Primitives."),
                ("Numéros des déclarations", "Supplémentaires."), 
                ("Date", "de l'expiration du délai de rectification."),
                ("Date", "de l'exigibilité des droits."),
                ("Numéros de la consignation des droits au sommier n' 28", " "),
                ("Recette des droits et amendes.", "Date"),
                ("Recette des droits et amendes.", "N^03"),
                ("Cautionnements. ", "Numéros de la consignation au sommier n'30"),
                ("Observations (les déclarations qui figurent à l'état n'413 doivent être émargées en conséquence, dans la présente colonne.)", " ")] 

        Some image (hence, some rows) may start with "Arrêté le \d{2} \w+ \d{4}( \w+)? servais" or contain notes.

    Task:
        Recognize the text from the image. Pay attention to reading each word and number correctly. Return the text as you read it and you must read the text from the image since the image contains texts.
    ```plaintext 
"""

## Few-shot

