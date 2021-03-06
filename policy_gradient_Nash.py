"""
<<<<<<< HEAD
Builds and trains two competing neural network that use policy gradients to learn to play Tic-Tac-Toe.

The input to the network is a vector with a number for each space on the board (One hot encoded so that it is a
vector of size 9 e.g. [0,0,0,0,1,0,-1,-1,1]). If the space has one of the networks
=======
Builds and trains a neural network that uses policy gradients to learn to play Tic-Tac-Toe.

The input to the network is a vector with a number for each space on the board. If the space has one of the networks
>>>>>>> cd7f1887036ae408030e1e8075ba553f93c7ab97
pieces then the input vector has the value 1. -1 for the opponents space and 0 for no piece.

The output of the network is a also of the size of the board with each number learning the probability that a move in
that space is the best move.

The network plays successive games randomly alternating between going first and second against an opponent that makes
moves by randomly selecting a free space. The neural network does NOT initially have any way of knowing what is or is not
<<<<<<< HEAD
a valid move, so initially it must learn the rules of the game. (This is a hyperparameter that can be switched though.)
=======
a valid move, so initially it must learn the rules of the game.
>>>>>>> cd7f1887036ae408030e1e8075ba553f93c7ab97

I have trained this version with success at 3x3 tic tac toe until it has a success rate in the region of 75% this maybe
as good as it can do, because 3x3 tic-tac-toe is a theoretical draw, so the random opponent will often get lucky and
force a draw.
"""
import functools
import numpy as np

from common.network_helpers import create_network_scope
from games.tic_tac_toe import TicTacToeGameSpec
from techniques.train_policy_gradient_Nash import train_policy_gradients

BATCH_SIZE = 100  # every how many games to do a parameter update?
LEARN_RATE = 1e-4
PRINT_RESULTS_EVERY_X = 1000  # every how many games to print the results
NETWORK_FILE_PATH = None#'current_network.p'  # path to save the network to
NUMBER_OF_GAMES_TO_RUN = 800000
COPY_NETWORK_AT = 0.65

# to play a different game change this to another spec, e.g TicTacToeXGameSpec or ConnectXGameSpec, to get these to run
# well may require tuning the hyper parameters a bit
game_spec = TicTacToeGameSpec()

create_network_func = functools.partial(create_network_scope, game_spec.board_squares(), (100, 100, 100), scope="player1") # Agent 1 learning network
create_network_func_2 = functools.partial(create_network_scope, game_spec.board_squares(), (100, 100, 100), scope="player2") # Agent 2 learning network


p1wins, p2wins, drawsarr = train_policy_gradients(game_spec, create_network_func, create_network_func_2, NETWORK_FILE_PATH,
                       number_of_games=NUMBER_OF_GAMES_TO_RUN,
                       batch_size=BATCH_SIZE,
                       learn_rate=LEARN_RATE,
                       print_results_every=PRINT_RESULTS_EVERY_X,
                       randomize_first_player=True,
                       copy_network_at=COPY_NETWORK_AT) #The first player always starts


np.save("pickles/train_policy_gradient_Nash_p1.npy", p1wins)
np.save("pickles/train_policy_gradient_Nash_p2.npy", p2wins)
np.save("pickles/train_policy_gradient_Nash_draws.npy", drawsarr)
print("saved!")