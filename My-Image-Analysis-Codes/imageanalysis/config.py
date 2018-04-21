import os
import configparser
import argparse

class CmdLineArguments:
    def __init__(self):
        parser = argparse.ArgumentParser() # creating an ArgumentParser object
        parser.add_argument("-p", "--path", help = "Path to data") # Add arguments:
        self.args = parser.parse_args()    # parses(=analyze) arguments

    def printOut(self):
        print("================")
        print("CmdLineArguments")
        print("================")
        print(str(self.args))
        print("")


class staticParameters:
    def __init__(self):
        """
        Create static parameter object with default parameters.
        """
        self.no_slices = 10
        self.no_time_steps = 59
        # we have to think about this. How to read/write
        self.image_dimensions = (512, 512) 

    def printOut(self):
        print("=================")
        print("StaticParameters:")
        print("=================")
        print("No. of Slices     = "+str(self.no_slices))
        print("No. of Time Steps = "+str(self.no_time_steps))
        print("Image Dimensions  = " + str(self.image_dimensions))
        print("")

    def writeToFile(self, fname):
        config = configparser.RawConfigParser()

        config.add_section('parameters')
        config.set('parameters', 'no_slices', self.no_slices)
        config.set('parameters', 'no_time_steps', self.no_time_steps)
        config.set('parameters', 'image_dimensions', self.image_dimensions)

        with open(fname, 'wb') as configfile:
            config.write(configfile)

    def readFromFile(self, fname):
        config = configparser.RawConfigParser()
        config.read(fname)
        self.no_slices = config.getint('parameters', 'no_slices')
        self.no_time_steps = config.getint('parameters', 'no_time_steps')
        self.image_dimensions = config.getint('parameters', 'image_dimensions')
