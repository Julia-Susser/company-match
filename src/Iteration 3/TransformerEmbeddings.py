from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer 
import torch.nn.functional as F 
import torch
import pandas as pd
from math import sqrt
model_ckpt = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_ckpt)


def sdp_attention(query, key, value):
    dim_k = query.size(-1) # dimension component
    sfact = sqrt(dim_k)     
    scores = torch.bmm(query, key.transpose(1,2)) / sfact
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

'''

Attention Class

# nn.linear : apply linear transformation to incoming data
#             y = x * A^T + b
# Ax = b where x is input, b is output, A is weight

# calculate scaled dot product attention matrix
# Requires embedding dimension 
# Each attention head is made of different q,k,v vectors

'''

class Attention(nn.Module):
    
    # initalisation 
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        
        # Define the three vectors
        # input - embed_dim, output - head_dim
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    # main class operation
    def forward(self, hidden_state):
        
        # calculate scaled dot product given a 
        attn_outputs = sdp_attention(
            self.q(hidden_state), 
            self.k(hidden_state), 
            self.v(hidden_state))
        
        return attn_outputs
    

    
'''

Multihead attention class

'''


class multiHeadAttention(nn.Module):
    
    # Config during initalisation
    def __init__(self, config):
        super().__init__()
        
        # model params, read from config file
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        
        # attention head (define only w/o hidden state)
        # each attention head is initialised with embedd/heads head dimension
        self.heads = nn.ModuleList(
            [Attention(embed_dim, head_dim) for _ in range(num_heads)])
        
        # output uses whole embedding dimension for output
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    # Given a hidden state (embeddings)
    # Apply operation for multihead attention
        
    def forward(self, hidden_state):
        
        # for each head embed_size/heads, calculate attention
        heads = [head(hidden_state) for head in self.heads] 
        x = torch.cat(heads, dim=-1) # merge/concat head data together
    
        # apply linear transformation to multihead attension scalar product
        x = self.out_linear(x)
        return x
    
    


class feedForward(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # define layer operations input x
        
    def forward(self, x):    # note must be forward
        x = self.gelu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
    
class encoderLayer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.attention = multiHeadAttention(config)    # multihead attention layer 
        self.feed_forward = feedForward(config)        # feed forward layer

    def forward(self, x):
        
        # Apply layer norm. to hidden state, copy input into query, key, value
        # Apply attention with a skip connection
        x = x + self.attention(self.norm1(x))
        
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.norm2(x))
        
        return x
    
    
'''

Token + Position Embedding 


'''

class tpEmbedding(nn.Module):
    
    def __init__(self, config):        
        super().__init__()
        
        # token embedding layer
        self.token_embeddings = nn.Embedding(config.vocab_size,
                                             config.hidden_size)
        
        # positional embedding layer
        # config.max_position_embeddings -> max number of positions in text 512 (tokens)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        
        # Create position IDs for input sequence
        seq_length = input_ids.size(1) # number of tokens
        position_ids = torch.arange(seq_length, dtype=torch.long)[None,:] # range(0,9)
        
        # tensor([[ 1996, 11286,  1997,  1037,  5340,  3392,  2003,  2200,  5931]])
        # tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
        
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Add normalisation & dropout layers
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    
    
# full transformer encoder combining the `Embedding` with the ``Embedding` ` layers

class TransformerEncoder(nn.Module):
    
    def __init__(self, config):       
        super().__init__()
        
        # token & positional embedding layer
        self.embeddings = tpEmbedding(config)
        
        # attention & forward feed layer 
        self.layers = nn.ModuleList([encoderLayer(config)
                                     for _ in range(config.num_hidden_layers)])
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, x):
        
        # embeddings layer output
        x = self.embeddings(x)
        
        # cycle through all heads
        for layer in self.layers:
            x = layer(x)
            
        x = x[:, 0, :] # select hidden state of [CLS] token
        x = self.dropout(x)
        return x
    

class TransformerEmbeddings():
    def __init__(self):
        self.model = TransformerEncoder(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        
    def getTensor(self,text):
        inputs = self.tokenizer(text, 
                   return_tensors="pt",      # pytorc tensor
                   add_special_tokens=False,
                          padding=True) # don't use pad, sep tokens
        return inputs.input_ids
    
    def train(self,data,epochs=1):
        tensorData = self.getTensor(data)
        for epoch in range(epochs):
            if epoch % 10 == 0:
                print("epoch %s" % epoch+1)
            self.model.forward(tensorData)
    
    def getEmbeddings(self,q):
        self.model.eval()
        inputs = self.getTensor(q)
        with torch.no_grad():
            x = self.model(inputs).cpu().detach().numpy()
            return x
        
# secDataPath = "../../inputs/SEC-CompanyTicker.csv"
# data = list(pd.read_csv(secDataPath,index_col=0).companyName[:100])
# train = TransformerEmbeddings()
# train.train(data)
