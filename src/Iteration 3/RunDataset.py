import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os


from TransformerEmbeddings import TransformerEmbeddings
from BERTEmbeddings import BERTEmbeddings
from OpenAIEmbeddings import OpenAIEmbeddings
from HNSW_FAISS import Faiss

secDataPath = "../../inputs/SEC-CompanyTicker.csv"
load_dotenv()
key = os.environ.get("OPENAI_KEY")



class runDataset():
    def __init__(self,model):
        models = ["Transformer", "BERT", "OpenAI"]
        self.model = model

        if ("data" not in model.keys()):
            raise Exception("No data inputted.")
        #not using word2vec because it performed terribly on datasets with only one or two words per code-word
        if (model["name"] == "Transformer"):
            self.Embeddings=TransformerEmbeddings()
            if ("epochs" not in model.keys()):
                model["epochs"] = 3
            #self.Embeddings.train(model["data"],epochs=model["epochs"])
        elif (model["name"] == "BERT"):
            self.Embeddings=BERTEmbeddings()
        elif (model["name"] == "OpenAI"):
            if ("api_key" not in model.keys()):
                raise Exception("No API Key inputted.")
            self.Embeddings = OpenAIEmbeddings(model["api_key"])
        else:
            raise Exception("Invalid Model")
        xb = self.Embeddings.getEmbeddings(model["data"])
        self.index = Faiss().faiss(xb)

    def getEmbeddings(self,x):
        return self.Embeddings.getEmbeddings(x)
    
    def similaritySearch(self,q,k=3):
        xq = self.getEmbeddings(q)
        D,I = Faiss().query(self.index,xq,k=k)
        I = I[0]
        guesses = [self.model["data"][guess] for guess in I]
        return guesses

# data = list(pd.read_csv(secDataPath,index_col=0).companyName[:1000])
# print(len(data))
# model = {
#     "name" : "Transformer",
#     "data" : data,
#     "epochs": 1,
#     "api_key":key
# }
# similaritySearch = runDataset(model)
# guesses = similaritySearch.similaritySearch(["Telekomunikasi Indonesia"])
# print(guesses)   


