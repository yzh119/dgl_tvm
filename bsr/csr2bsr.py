from scipy import sparse
import numpy as np

row = np.array([0, 1, 0, 1, 0, 2, 3, 2, 3])
col = np.array([0, 0, 1, 1, 2, 2, 2, 3, 3])
data = np.ones(9)

spmat = sparse.csr_matrix((data, (row, col)), shape=(4, 4))
print(spmat)
spmat_bsr = spmat.tobsr()
print(spmat_bsr.indptr, spmat_bsr.indices, spmat_bsr.data)
