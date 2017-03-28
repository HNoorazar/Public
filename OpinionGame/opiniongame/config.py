"""
Simulation configuration related code goes in here.abs

- Configuration reader and state object.
- Data structure containing a set of closures for specific functions
  needed by the simulation that we want to make parameters.
- command line argument processing

"""

import configparser
import argparse

class CmdLineArguments:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", help =    "configuration file.")
        parser.add_argument("-a", "--adjacency", help = "adjacency matrix of graph.")
        parser.add_argument("-w", "--weights", help =   "Topics Coupling Weight.")
        parser.add_argument("-i", "--initialOpinions", help = "Initial Opinions Matrix.")
        self.args = parser.parse_args()

    def printOut(self):
        print("================")
        print("CmdLineArguments")
        print("================")
        print(str(self.args))
        print("")

class UserFunctions:
    def __init__(self, selectingMethod, stoppingMethod, dPotential):
        self.selector = selectingMethod
        self.stop = stoppingMethod
        self.dPotential = dPotential

class staticParameters:
    def __init__(self):
        """
        Create static parameter object with default parameters.
        """
        self.learning_rate = .1
        self.no_of_experiments = 100
        self.popSize    = 20
        self.threshold  = .000001
        self.Hthreshold = .000001
        self.Kthreshold = .000001 * self.popSize
        self.ntopics = 1
        self.startingseed = 10
        self.iterationMax = 100

    def printOut(self):
        print("=================")
        print("StaticParameters:")
        print("=================")
        print("Learning rate  = "+str(self.learning_rate))
        print("NumExperiments = "+str(self.no_of_experiments))
        print("PopSize        = "+str(self.popSize))
        print("Threshold      = "+str(self.threshold))
        print("Hthreshold     = "+str(self.Hthreshold))
        print("Kthreshold     = "+str(self.Kthreshold))
        print("NTopics        = "+str(self.ntopics))
        print("startingSeed   = "+str(self.startingseed))
        print("iterationMax   = "+str(self.iterationMax))
        print("")

    def writeToFile(self, fname):
        config = configparser.RawConfigParser()

        config.add_section('parameters')
        config.set('parameters', 'learning_rate', self.learning_rate)
        config.set('parameters', 'threshold',  self.threshold)
        config.set('parameters', 'Hthreshold', self.Hthreshold)
        config.set('parameters', 'Kthreshold', self.Kthreshold)
        config.set('parameters', 'ntopics',    self.ntopics)
        config.set('parameters', 'popSize',    self.popSize)
        config.set('parameters', 'no_of_experiments', self.no_of_experiments)
        config.set('parameters', 'startingseed', self.startingseed)
        config.set('parameters', 'iterationMax', self.iterationMax)

        with open(fname, 'wb') as configfile:
            config.write(configfile)

    def readFromFile(self, fname):
        config = configparser.RawConfigParser()
        config.read(fname)

        self.learning_rate = config.getfloat('parameters', 'learning_rate')
        self.threshold     = config.getfloat('parameters', 'threshold')
        self.Hthreshold    = config.getfloat('parameters', 'Hthreshold')
        self.Kthreshold    = config.getfloat('parameters', 'Kthreshold')
        self.ntopics       = config.getint('parameters', 'ntopics')
        self.popSize       = config.getint('parameters', 'popSize')
        self.no_of_experiments = config.getint('parameters', 'no_of_experiments')
        self.startingseed  = config.getint('parameters', 'startingseed')
        self.iterationMax  = config.getint('parameters', 'iterationMax')
