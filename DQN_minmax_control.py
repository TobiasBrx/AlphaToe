"""
IMPORTANT:
    RESTART KERNEL BEFORE RUNNING. OTHERWISE NETWORK COPY DOESN'T WORK.
Builds and trains two competing neural network that use DQN to learn to play Tic-Tac-Toe.

The input to the network is a vector with a number for each space on the board (One hot encoded so that it is a
vector of size 9 e.g. [0,0,0,0,1,0,-1,-1,1]). If the space has one of the networks
pieces then the input vector has the value 1. -1 for the opponents space and 0 for no piece.

The output of the network is a also of the size of the board with each number learning the probability that a move in
that space is the best move.

The network plays successive games randomly alternating between going first and second against an opponent that makes
moves by randomly selecting a free space. The neural network does initially know what is or is not
a valid move, so initially it does not have to learn the rules of the game. (This is a hyperparameter that can be switched though.)

Base version is independent Networks:
Training is unstable here if we leave the two networks independent. If we fix the first player, then this player
will generally have an advantage over the second player and on average perform better.
Nevertheless, the second player might get the upper hand for some time (many thousand episodes).
Interestingly they hardly ever draw in this version. 
The players do not manage to learn the game. 
Q-values do not diverge. Winning rates shift significantly untill one agent dies in a minimum.

I will now implement the option of copying over the winning network to the other.

"""
###############################   Imports    ##################################

import functools
from common.network_helpers import create_network_scope
from games.tic_tac_toe import TicTacToeGameSpec
from techniques.DQN_minmax import DQN_train_Nash
from techniques.min_max import min_max_alpha_beta

###############################################################################

BATCH_SIZE = 100  # every how many games to do a parameter update?
LEARN_RATE = 1e-4
PRINT_RESULTS_EVERY_X = 1000  # every how many games to print the results
NETWORK_FILE_PATH = "pickles/DQN_Nash_well_trained_Network_Pickle"#/'current_network.p'  # path to save the network to
NUMBER_OF_GAMES_TO_RUN = 500000
COPY_NETWORK_AT = 0.55 # winning rate after which the network is copied over
network_file_path_load = None # "pickles/DQN_Nash_well_trained_Network_Pickle"
# to play a different game change this to another spec, e.g TicTacToeXGameSpec or ConnectXGameSpec, to get these to run
# well may require tuning the hyper parameters a bit
game_spec = TicTacToeGameSpec()

# creating the competing networks
create_network_func = functools.partial(create_network_scope, game_spec.board_squares(),\
                                        (100, 100, 100), output_softmax=False, scope="principal")
create_network_func_2 = functools.partial(create_network_scope, game_spec.board_squares(),\
                                        (100, 100, 100), output_softmax=False, scope="principal_2")
create_network_func_target = functools.partial(create_network_scope, game_spec.board_squares(),\
                                        (100, 100, 100), output_softmax=False, scope="target")
create_network_func_target_2 = functools.partial(create_network_scope, game_spec.board_squares(),\
                                        (100, 100, 100), output_softmax=False, scope="target_2")

def min_max_move_func(board_state, side):
    return min_max_alpha_beta(game_spec, board_state, side, 5)[1]


DQN_train_Nash(game_spec, create_network_func, create_network_func_2, 
                      create_network_func_target, create_network_func_target_2,
                      network_file_path=network_file_path_load,
                      save_network_file_path=NETWORK_FILE_PATH,
                      opponent_func=min_max_move_func,
                      number_of_games=NUMBER_OF_GAMES_TO_RUN,
                      batch_size=BATCH_SIZE,
                      learn_rate=LEARN_RATE,
                      print_results_every=PRINT_RESULTS_EVERY_X,
                      randomize_first_player=True,
                      copy_network_at=COPY_NETWORK_AT)
