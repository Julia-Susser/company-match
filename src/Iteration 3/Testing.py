import unittest
import pandas as pd
from dotenv import load_dotenv
import os

from RunDataset import runDataset

secDataPath = "../../inputs/SEC-CompanyTicker.csv"
load_dotenv()
key = os.environ.get("OPENAI_KEY")

class RunDatasetTest(unittest.TestCase):
   
    def getModel(self,data):
        return {
            "name" : "Transformer",
            "data" : data,
            "epochs": 1,
            "api_key":key
        }
    
    def setUp(self):

        secDataPath = "../../inputs/SEC-CompanyTicker.csv"
        self.secData = list(pd.read_csv(secDataPath,index_col=0).companyName[:100])
        self.secModel = self.getModel(self.secData)
        self.SECSearch = runDataset(self.secModel)

    def test1(self):
        solution = "Shell Plc"
        result = self.SECSearch.similaritySearch(["Shell Plc"],k=3)   
        self.assertIn(solution, result)

    def test2(self):
        # print(self.secData[24])
        solution = "Chevron Corp"
        result = self.SECSearch.similaritySearch(["Chevron"],k=3) 
        print(result)
        self.assertIn(solution, result)



if __name__ == '__main__':
    unittest.main()
