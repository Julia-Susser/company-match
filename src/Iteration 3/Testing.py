import unittest
import pandas as pd
from dotenv import load_dotenv
import os

from RunDataset import runDataset

secDataPath = "../../inputs/SEC-CompanyTicker.csv"
load_dotenv()
key = os.environ.get("OPENAI_KEY")

class RunDatasetTest(unittest.TestCase):
   
    def getModel(self,data,epochs):
        return {
            "name" : "Transformer",
            "data" : data,
            "epochs": epochs,
            "api_key":key
        }
    
    def setUp(self):
        pass
        
    # def test1(self):
    #     secDataPath = "../../inputs/SEC-CompanyTicker.csv"
    #     self.secData = list(pd.read_csv(secDataPath,index_col=0).companyName[:100])
    #     self.secModel = self.getModel(self.secData,epochs=100)
    #     self.SECSearch = runDataset(self.secModel)

    #     solution = "Shell Plc"
    #     result = self.SECSearch.similaritySearch(["Shell"],k=3)   
    #     print(result)
    #     self.assertIn(solution, result)

    #     solution = "Chevron Corp"
    #     result = self.SECSearch.similaritySearch(["Chevron"],k=3) 
    #     print(result)
    #     self.assertIn(solution, result)


    def test2(self):
        secDataPath = "../../inputs/SEC-CompanyTicker.csv"
        self.secData = list(pd.read_csv(secDataPath,index_col=0).companyName[:300])
        self.secModel = self.getModel(self.secData, epochs=20)
        self.SECSearch = runDataset(self.secModel)

        solution = "Shell Plc"
        result = self.SECSearch.similaritySearch(["shell"],k=3)   
        print(result)
        self.assertIn(solution, result)

        solution = "Chevron Corp"
        result = self.SECSearch.similaritySearch(["Chevron"],k=3) 
        print(result)
        self.assertIn(solution, result)


if __name__ == '__main__':
    unittest.main()
