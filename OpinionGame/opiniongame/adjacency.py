import numpy as np

def make_adj(pop_size, topology):
    """
    Create an adjacency matrix for a few specific types of graph given a population
    size and topology.  Supported types: star, full, grid.
    """

    adj = None

    if topology == 'star':
        adj = np.zeros((pop_size, pop_size))
        adj[:, [-1]] = 1
        adj[[-1], :] = 1
        adj[-1, -1] = 0
    elif topology == 'full':
        adj = np.zeros((pop_size, pop_size))
        adj = np.ones((pop_size, pop_size)) - np.identity(pop_size)
    elif topology == 'grid':
        adj = generateGridAdj(pop_size)
    return adj

#
# generate a grid graph.  computes the dimensions for the grid by
# calculating the prime factors (with duplication) of the population
# size and lets the nrows be the product of the odd elements of the
# factor list, and the ncols be the product of the even elements of
# the factor list.
#
def generateGridAdj(pop_size):
    def primes(n):
        primfac = []
        d = 2
        while d*d <= n:
            while (n % d) == 0:
                primfac.append(d)  # supposing you want multiple factors repeated
                n //= d
            d += 1
        if n > 1:
            primfac.append(n)
        return primfac

    factors = primes(pop_size)
    nrows = np.prod(factors[0::2])
    ncols = np.prod(factors[1::2])

    adj = np.zeros((nrows * ncols, nrows * ncols))

    for i in range(nrows):
        for j in range(ncols):
            idx = i*ncols + j
            if i-1 >= 0:
                nidx = (i-1)*ncols + j
                adj[idx, nidx] = 1
                adj[nidx, idx] = 1
            if i+1 < nrows:
                nidx = (i+1)*ncols + j
                adj[idx, nidx] = 1
                adj[nidx, idx] = 1
            if j-1 >= 0:
                nidx = i*ncols + (j-1)
                adj[idx, nidx] = 1
                adj[nidx, idx] = 1
            if j+1 < ncols:
                nidx = i*ncols + (j+1)
                adj[idx, nidx] = 1
                adj[nidx, idx] = 1

    return adj
