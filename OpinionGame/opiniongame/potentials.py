"""
Potential function derivative definitions.

In this code potentials are defined as closures.
This allows one to instantate a parameterized potential once and pass around the
resulting function as a closure without needing to thread the potential parameters
through the code.
"""
from math import exp, pi

#
# equation for tent:
#
# x/tau                   0.0 <= x <= tau
# -x/(1-tau) + 1/(1-tau)  tau < x <= 1.0
# 0.0                     elsewhere
#
def createTent(center):
    assert center >= 0.0 and center <= 1.0
    def tent(x):
        if x >= 0 and x < center:
            return 1.0 / center
        elif x >= center and x <= 1.0:
            return -1.0 / (1.0 - center)
        else:
            return 0
    return tent

def flatPotential(x):
    return 0.0

def flatTopTent(flatRegion):
    assert len(flatRegion) == 2
    assert flatRegion[0] < flatRegion[1]
    assert flatRegion[0] >= 0.0 and flatRegion[0] <= 1.0
    assert flatRegion[1] >= 0.0 and flatRegion[1] <= 1.0

    def flatTent(x):
        if x >= 0 and x < flatRegion[0]:
            return 1.0 / flatRegion[0]
        elif x >= flatRegion[0] and x < flatRegion[1]:
            return 0.0
        elif x >= flatRegion[1] and x <= 1:
            return -1.0 / (1.0 - flatRegion[1])
        else:
            return 0.0

    return flatTent

def gaussian(x):
    mean = .5
    z = (2 * pi)**(.5)
    if x >= 0 and x <= 1:
        return ((mean - x)/z) * exp(-(x - mean)**2/2.0)
    else:
        return 0.0
        