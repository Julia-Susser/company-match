from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
#word2vec is bad for datasets with only one or two words because it doesnt not look for co occurences between different inputs
#Since it only looks at the words its surrounded by in its input, it lacks the same self attention features as other models
#Indeed, the model looks at the surrounding words to predict the target word (Skip-gram) or predicts surrounding words given the target word (CBOW).
# so if the input is one word, it doesn't really work, much better for longer sentences

import time
from gensim.models import Word2Vec
from tqdm import tqdm

data = list(pd.read_csv(secDataPath, index_col=0).companyName[:100]) + ["shell"]

tqdm.pandas()
def preprocessing(titles_array):
    processed_array = []
    for title in tqdm(titles_array):
        # remove other non-alphabets symbols with space (i.e. keep only alphabets and whitespaces).
        processed = re.sub('[^a-zA-Z ]', '', title)
        words = processed.split()
        # keep words that have length of more than 1 (e.g. gb, bb), remove those with length 1.
        processed_array.append(' '.join([word for word in words if len(word) > 1]))
    return processed_array



# Assuming 'processed' is a list of strings
data = list(pd.read_csv(secDataPath, index_col=0).companyName[:100]) + ["shell"]
processed = preprocessing(data)

# Use tqdm's progress_apply with lambda function
data = [sublist[0] for sublist in tqdm(pd.DataFrame({"companyName": processed}).apply(lambda x: x.str.split()).values)]



class Word2VecEmbeddings():
    # Required to train on all data and queries because use keys to find embeddings
    def train(self, data, epochs=100):
#         # Convert data to a list of lists
#         data = [[x] for x in data]
        
        # Initialize Word2Vec model
        self.model = Word2Vec(data, 
                              min_count=1, 
                              vector_size=768,
                              window=5, 
                              sg=1)
        
        # Train the model
        for epoch in range(epochs):
            if epoch % 10 == 0:
                print("epoch %s" % epoch)
            self.model.train(data, total_examples=self.model.corpus_count, epochs=100)
            
    def getEmbeddings(self, q):
        values = []
        maxlength = 0

        # Find the maximum length of sequences
        for sentence in q:
            embedding = []
            for val in sentence:
                embedding += list(self.model.wv.get_vector(val))
            values.append(embedding)
            maxlength = max(maxlength, len(embedding))

        # Pad each sequence individually
        padded_values = pad_sequences(values, maxlen=3840, padding='post', dtype='float32')

        return np.array(padded_values)

    
    def getKeys(self):
        words = list(self.model.wv.index_to_key)
        return words



# Create an instance of Word2VecEmbeddings
train = Word2VecEmbeddings()

# Train the model with the data
train.train(data)

# Get embeddings for the query and the full data
xq = train.getEmbeddings([["shell"]])
xb = train.getEmbeddings(data)

# Now xq should work as expected
