# test driver to verify that new version of code works
import opiniongame.config as og_cfg
import opiniongame.IO as og_io
import opiniongame.coupling as og_coupling
import opiniongame.state as og_state
import opiniongame.opinions as og_opinions
import opiniongame.adjacency as og_adj
import opiniongame.selection as og_select
import opiniongame.potentials as og_pot
import opiniongame.core as og_core
import opiniongame.stopping as og_stop
import numpy as np

#
# process command line
#
cmdline = og_cfg.CmdLineArguments()
cmdline.printOut()

#
# load configuration
#
# TODO: add option to generate defaults and save to file
# TODO: interpret args to get filename if specified on cmd line
config = og_cfg.staticParameters()
config.readFromFile('staticParameters.cfg')
config.threshold = 0.0001
config.Kthreshold = 0.00001
config.printOut()

#
# seed PRNG: must do this before any random numbers are
# ever sampled during default generation
#
print("SEEDING PRNG: "+str(config.startingseed))
np.random.seed(config.startingseed)

state = og_state.WorldState.fromCmdlineArguments(cmdline, config)

#
# run
#

config.learning_rate = 0.01
tau = 0.66

ufuncs = og_cfg.UserFunctions(og_select.FastPairSelection,
                              og_stop.totalChangeStop,
                              og_pot.createTent(tau))
state = og_core.run_until_convergence(config, state, ufuncs)

rdict = {}
rdict['history'] = state.history
og_io.saveMatrix('output.mat', rdict)