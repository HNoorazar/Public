import numpy as np

def initialize_opinions(pop_size, ntopics):
    """
    Given a population size and topic count, generate a set of random
    opinions.  Assumes each topic space is the scalars over [0,1].
    """
    return np.random.rand(pop_size, ntopics)

def isPolarized(opinions, threshold):
    """
    Return a list of polarization determinations for a given 
    set of opinions.  The determination can be either true, false,
    or none (if it can't be determined).
    """
    ntopics = np.shape(opinions)[1]

    polarization = []
    for topic in range(ntopics):
        onTopic = opinions[:, topic]
        onTopic.sort()
        if onTopic[-1]-onTopic[0] < threshold:
            polarization.append(False)
        elif 1.0 - (onTopic[-1]-onTopic[0]) < threshold:
            polarization.append(True)
        else:
            polarization.append(None)
    return polarization
