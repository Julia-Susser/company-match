from openai import OpenAI
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()
key = os.environ.get("OPENAI_KEY")

# Initialize OpenAI client (replace '...' with your API key)
client = OpenAI(api_key=key)


class OpenAIEmbeddings:
    def __init__(self,api_key):
        self.client = OpenAI(api_key=api_key)
        
    
    def getEmbeddings(self, text_list):
        data = self.client.embeddings.create(input=text_list, model='text-embedding-ada-002').data
        embeddings = [embedding.embedding for embedding in data]
        return np.array(embeddings)
    
    
train = OpenAIEmbeddings(key)
