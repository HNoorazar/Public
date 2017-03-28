"""
Object containing dynamic simulation state.
"""

import opiniongame.IO as og_io
import opiniongame.coupling as og_coupling
import opiniongame.opinions as og_opinions
import opiniongame.adjacency as og_adj
import numpy as np

class WorldState:
    def __init__(self, adj, couplingWeights, initialOpinions):
        self.adj = adj
        self.couplingWeights = couplingWeights
        self.initialOpinions = initialOpinions
        self.history = None
        self.iterCount = 0

    @classmethod
    def fromCmdlineArguments(cls, cmdline, config):
        #
        # check optional arguments and generate defaults if missing
        #
        weights = None
        initialOpinions = None
        adj = None

        if cmdline.args.weights is not None:
            weights = og_io.loadNamedMatrix(cmdline.args.weights, 'weights')
        else:
            weights = og_coupling.weights_no_coupling(config.popSize, config.ntopics)

        if cmdline.args.initialOpinions is not None:
            initialOpinions = og_io.loadNamedMatrix(cmdline.args.initialOpinions, 'initialOpinions')
        else:
            initialOpinions = og_opinions.initialize_opinions(config.popSize, config.ntopics)

        if cmdline.args.adjacency is not None:
            adj = og_io.loadNamedMatrix(cmdline.args.adjacency, 'adjacency')
        else:
            adj = og_adj.make_adj(config.popSize, 'full')
        
        state = cls(adj, weights, initialOpinions)
        state.validate()

        #
        # set popsize and ntopics based on current state.  warn if config 
        # disagrees with loaded files.
        #
        wPopsize = np.shape(weights)[0]
        wNtopics = np.shape(weights)[1]

        if wPopsize != config.popSize:
            print("WARNING: popsize from data files disagrees with cfg.")
            config.popSize = wPopsize

        if wNtopics != config.ntopics:
            print("WARNING: ntopics from data files disagrees with cfg.")
            config.ntopics = wNtopics

        return state


    def initializeHistory(self):
        hist = np.array([self.initialOpinions], copy=True)
        hist = np.concatenate((hist, [self.initialOpinions]), axis=0)
        self.history = hist

    def appendToHistory(self, newOpinions):
        self.history = np.concatenate((self.history, [newOpinions]), axis=0)

    def reset(self):
        self.history = None
        self.iterCount = 0

    def currentOpinions(self):
        return self.history[-1]

    def validate(self):
        # validation of data sizes
        print("WEIGHT SHAPE   : "+str(np.shape(self.couplingWeights)))
        print("OPINION SHAPE  : "+str(np.shape(self.initialOpinions)))
        print("ADJACENCY SHAPE: "+str(np.shape(self.adj)))

        wPopsize = np.shape(self.couplingWeights)[0]
        wNtopics1 = np.shape(self.couplingWeights)[1]
        wNtopics2 = np.shape(self.couplingWeights)[2]

        oPopsize = np.shape(self.initialOpinions)[0]
        oNtopics = np.shape(self.initialOpinions)[1]

        aPopsize1 = np.shape(self.adj)[0]
        aPopsize2 = np.shape(self.adj)[1]

        if aPopsize1 != aPopsize2:
            raise ValueError("Adjacency matrix must be square.")
        if wNtopics1 != wNtopics2:
            raise ValueError("Per-topic weight matrix must be square.")
        if wPopsize != oPopsize or wPopsize != aPopsize1 or aPopsize1 != oPopsize:
            raise ValueError("Weight tensor, opinion state, and adjacency matrix disagree on population size.")
        if oNtopics != wNtopics1:
            raise ValueError("Weight tensor and opinion state disagree on topic count.")

        print("==> World state validation passed.")
        print("")
