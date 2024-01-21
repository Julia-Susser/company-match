import faiss
    
class Faiss:
    def __init__(self):
        pass

    def faiss(self,xb):
        d = xb[0].size
        M = 32
        index = faiss.IndexHNSWFlat(d, M)            
        index.hnsw.efConstruction = 40         # Setting the value for efConstruction.
        index.hnsw.efSearch = 16               # Setting the value for efSearch.
        index.add(xb)
        return index
    
    def query(self,index,xq,k=3):
        D, I = index.search(xq, k)   
        return D, I

