import numpy as np
import faiss      


d = 512                           # dimension
nb = 8000                      # database size
nq = 5                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')

xq = np.random.random((nq, d)).astype('float32')


index = faiss.IndexFlatL2(d)   # build the index
# print(index.is_trained)
index.add(xb)                  # add vectors to the index
# print(index.ntotal)

import time 
for i in range(100):
    st = time.time()
    k = 4                          # we want to see 4 nearest neighbors
    D, I = index.search(xq, 5) # sanity check
    # print(I)
    # print(D)
    # print(time.time()-st)
