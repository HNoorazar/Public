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
config.threshold = 0.01
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
tau_list = np.arange(0.45, 0.9, 0.01)
alpha_list = np.arange(0.05, 0.25, 0.01)
numalphas = len(alpha_list)
numtaus = len(tau_list)

numvars = 3

resultMatrix = np.zeros((numalphas, numtaus, numvars))

for (i, alpha) in enumerate(alpha_list):
    config.learning_rate = alpha
    print("")

    for (j, tau) in enumerate(tau_list):
        print((alpha, tau))
        #
        # functions for use by the simulation engine
        #
        ufuncs = og_cfg.UserFunctions(og_select.FastPairSelection,
                                      og_stop.totalChangeStop,
                                      og_pot.createTent(tau))

        polarized = 0
        notPolarized = 0
        aveIters = 0
        for k in range(100):
            state = og_core.run_until_convergence(config, state, ufuncs)

            results = og_opinions.isPolarized(state.history[-1], 0.05)
            for result in results:
                 if result:
                     polarized += 1
                 else:
                     notPolarized += 1
            aveIters += state.iterCount
            state.reset()
            state.initialOpinions = og_opinions.initialize_opinions(config.popSize, config.ntopics)

        resultMatrix[i][j][0] = polarized
        resultMatrix[i][j][1] = notPolarized
        resultMatrix[i][j][2] = aveIters/100.0

rdict = {}
rdict['results'] = resultMatrix
og_io.saveMatrix('output.mat', rdict)
