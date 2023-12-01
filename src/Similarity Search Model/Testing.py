import unittest
import pandas as pd
from dotenv import load_dotenv
import os

from RunDataset import runDataset

secDataPath = "../../inputs/SEC-CompanyTicker.csv"
load_dotenv()
key = os.environ.get("OPENAI_KEY")

class RunDatasetTest(unittest.TestCase):
   
    def getModel(self,name,data,epochs=1):
        return {
            "name" : name,
            "data" : data,
            "epochs": epochs,
            "api_key":key
        }
    
    def setUp(self):
        pass
        
    def test1(self):
        secDataPath = "../../inputs/SEC-CompanyTicker.csv"
        self.secData = list(pd.read_csv(secDataPath,index_col=0).companyName[:100])
        self.secModel = self.getModel("Transformer",self.secData,epochs=100)
        self.SECSearch = runDataset(self.secModel)

        solution = "Shell Plc"
        result = self.SECSearch.similaritySearch(["Shell"],k=3)   
        print(result)
        self.assertIn(solution, result)

        solution = "Chevron Corp"
        result = self.SECSearch.similaritySearch(["Chevron"],k=3) 
        print(result)
        self.assertIn(solution, result)


    def test2(self):
        secDataPath = "../../inputs/SEC-CompanyTicker.csv"
        self.secData = list(pd.read_csv(secDataPath,index_col=0).companyName[:300])
        self.secModel = self.getModel("Transformer",self.secData, epochs=20)
        self.SECSearch = runDataset(self.secModel)

        solution = "Microsoft Corp"
        result = self.SECSearch.similaritySearch(["microsoft"],k=3)   
        print(result)
        self.assertIn(solution, result)

        solution = "Nvidia Corp"
        result = self.SECSearch.similaritySearch(["nvidia"],k=3) 
        print(result)
        self.assertIn(solution, result)

    def testOpenAI(self):
        secDataPath = "../../inputs/SEC-CompanyTicker.csv"
        self.secData = list(pd.read_csv(secDataPath,index_col=0).companyName[:300])
        self.secModel = self.getModel("OpenAI",self.secData)
        self.SECSearch = runDataset(self.secModel)

        solution = "Microsoft Corp"
        result = self.SECSearch.similaritySearch(["microsoft"],k=3)   
        print(result)
        self.assertIn(solution, result)

        solution = "Nvidia Corp"
        result = self.SECSearch.similaritySearch(["nvidia"],k=3) 
        print(result)
        self.assertIn(solution, result)

    def testBERT(self):
        secDataPath = "../../inputs/SEC-CompanyTicker.csv"
        self.secData = list(pd.read_csv(secDataPath,index_col=0).companyName[:300])
        self.secModel = self.getModel("BERT",self.secData)
        self.SECSearch = runDataset(self.secModel)

        solution = "Microsoft Corp"
        result = self.SECSearch.similaritySearch(["microsoft"],k=3)   
        print(result)
        self.assertIn(solution, result)

        solution = "Nvidia Corp"
        result = self.SECSearch.similaritySearch(["nvidia"],k=3) 
        print(result)
        self.assertIn(solution, result)


if __name__ == '__main__':
    testAll = True
    if testAll:
        unittest.main() 
    else:
        # Specify the test method names to run
        test_names_to_run = ['testBERT']

        # Load the test suite with the specified test method names
        suite = unittest.TestLoader().loadTestsFromNames(test_names_to_run, RunDatasetTest)

        # Run the test suite
        unittest.TextTestRunner().run(suite)
