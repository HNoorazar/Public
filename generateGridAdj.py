import numpy as np

##
## nrows (and ncols) are number of rows (and cols)in the grid graph
##
def generateGridAdj(nrows, ncols):
    Adj = np.zeros((nrows * ncols,nrows * ncols))
    
    
    for i in range(ncols):
        for j in range(nrows):
            k = np.ravel_multi_index((i,j), dims=(ncols, nrows), order='F')
            if i > 0:
                ii = i-1
                jj = j
                kk = np.ravel_multi_index((ii,jj), dims=(ncols, nrows), order='F')
                Adj[k,kk] = 1
                        
            if i<ncols-1:
                ii=i+1
                jj=j
                kk = np.ravel_multi_index((ii,jj), dims=(ncols, nrows), order='F')
                Adj[k,kk] = 1
            
            if j>0:
                ii=i
                jj=j-1
                kk = np.ravel_multi_index((ii,jj), dims=(ncols, nrows), order='F')
                Adj[k,kk] = 1
            if j<nrows-1:
                ii=i
                jj=j+1
                kk = np.ravel_multi_index((ii,jj), dims=(ncols, nrows), order='F')
                Adj[k,kk] = 1
        
    return Adj