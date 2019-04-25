#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:23:11 2019

@author: tobiasbraun
"""

"""
Builds and trains a neural network that uses TD learning to learn to play Tic-Tac-Toe.

The input to the network is a vector with a number for each space on the board (One hot encoded so that it is a
vector of size 9 e.g. [0,0,0,0,1,0,-1,-1,1]). If the space has one of the networks
pieces then the input vector has the value 1. -1 for the opponents space and 0 for no piece.

The output of the network is a also of the size of the board with each number learning the probability that a move in
that space is the best move.

The network plays successive games randomly alternating between going first and second against an opponent that makes
moves by randomly selecting a free space. The neural network does initially know what is or is not
a valid move, so initially it does not have to learn the rules of the game.(This is a hyperparameter that can be switched though.)

Q-values will diverge here and learning is unstable. Deadly Triad.
"""
import functools

from common.network_helpers import create_network
from games.tic_tac_toe import TicTacToeGameSpec
from techniques.td_learning import td_learning_train

BATCH_SIZE = 100  # every how many games to do a parameter update?
LEARN_RATE = 1e-4
PRINT_RESULTS_EVERY_X = 1000  # every how many games to print the results
NETWORK_FILE_PATH = None#'current_network.p'  # path to save the network to
NUMBER_OF_GAMES_TO_RUN = 200000

# to play a different game change this to another spec, e.g TicTacToeXGameSpec or ConnectXGameSpec, to get these to run
# well may require tuning the hyper parameters a bit
game_spec = TicTacToeGameSpec()

create_network_func = functools.partial(create_network, game_spec.board_squares(), (100, 100, 100), output_softmax=False)
create_network_func_2 = functools.partial(create_network, game_spec.board_squares(), (100, 100, 100), output_softmax=False)



td_learning_train(game_spec, create_network_func, create_network_func_2, NETWORK_FILE_PATH,
                       number_of_games=NUMBER_OF_GAMES_TO_RUN,
                       batch_size=BATCH_SIZE,
                       learn_rate=LEARN_RATE,
                       print_results_every=PRINT_RESULTS_EVERY_X,
                       randomize_first_player=True) # randomized first player
