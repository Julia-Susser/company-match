from transformers import AutoTokenizer, AutoModel
import torch

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BERTEmbeddings:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)
        
    #CLS is a special classification token and the last hidden state of BERT Embedding
    def cls_pooling(self, model_output):
        return model_output.last_hidden_state[:, 0]

    #BERT tokenizer of input text
    def getEmbeddings(self, text_list):
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return self.cls_pooling(model_output).cpu().detach().numpy()
    
# train = BERTEmbeddings()
# train.getEmbeddings(["julia"])