#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:20:28 2019

@author: tobiasbraun

This is a prime example of the Deadly Triad problem from the Deepmind paper by van Hasselt et al.
THe Q-values diverge because of the risky combination of off-policy learning, bootstrapping and 
function approximation. It actually goes as far as underperforming against a random player. 
This means it actually learns to play badly.

"""
###############################   Imports    ##################################
import collections
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from common.network_helpers import load_network, save_network

###############################################################################

def get_td_network_move(session, input_layer, output_layer, board_state, side, eps=0.1,
                                valid_only=False, game_spec=None, ):
    """Choose a move for the given board_state using a stocastic policy. A move is selected using the values from the
    output_layer as a categorical probability distribution to select a single move
    
    Args:
        session (tf.Session): Session used to run this network
        input_layer (tf.Placeholder): Placeholder to the network used to feed in the board_state
        output_layer (tf.Tensor): Tensor that will output the probabilities of the moves, we expect this to be of
        dimesensions (None, board_squares) and the sum of values across the board_squares to be 1.
        board_state: The board_state we want to get the move for.
        side: The side that is making the move.
    
    Returns:
        (np.array) It's shape is (board_squares), and it is a 1 hot encoding for the move the network has chosen.
        """
    np_board_state = np.array(board_state)
    if side == -1:
        np_board_state = -np_board_state
    
    np_board_state = np_board_state.reshape(1, *input_layer.get_shape().as_list()[1:])
    Q_values_of_actions = session.run(output_layer,
                                      feed_dict={input_layer: np_board_state})[0]
    
    if valid_only:
        available_moves = list(game_spec.available_moves(board_state))
        #print(available_moves)
        
        if len(available_moves) == 1:
            move = np.zeros(game_spec.board_squares())
            np.put(move, game_spec.tuple_move_to_flat(available_moves[0]), 1)
            #print(move)
            return move
        available_moves_flat = [game_spec.tuple_move_to_flat(x) for x in available_moves]
        
        if np.random.rand() < eps:
            pick = random.choice(available_moves_flat)
            move = np.zeros(game_spec.board_squares())
            np.put(move, pick, 1)
            #print(move)
            return move
        for i in range(game_spec.board_squares()):
            if i not in available_moves_flat:
                Q_values_of_actions[i] = - np.inf
            
        pick = np.argmax(Q_values_of_actions)
        best_move = np.zeros(game_spec.board_squares())
        np.put(best_move, pick, 1)
        return best_move
        
    else:
        pick = np.argmax(Q_values_of_actions)
        best_move = np.zeros(game_spec.board_squares())
        np.put(best_move, pick, 1)
    return best_move

###############################################################################

def td_learning_train(game_spec,
                           create_network,
                           create_network_2,
                           network_file_path,
                           save_network_file_path=None,
                           opponent_func=None,
                           number_of_games=10000,
                           print_results_every=1000,
                           learn_rate=1e-4,
                           batch_size=100,
                           randomize_first_player=True):
    """Train a network using temproal difference learning

    Args:
        save_network_file_path (str): Optionally specifiy a path to use for saving the network, if unset then
            the network_file_path param is used.
        opponent_func (board_state, side) -> move: Function for the opponent, if unset we use an opponent playing
            randomly
        randomize_first_player (bool): If True we alternate between being the first and second player
        game_spec (games.base_game_spec.BaseGameSpec): The game we are playing
        create_network (->(input_layer : tf.placeholder, output_layer : tf.placeholder, variables : [tf.Variable])):
            Method that creates the network we will train.
        network_file_path (str): path to the file with weights we want to load for this network
        number_of_games (int): number of games to play before stopping
        print_results_every (int): Prints results to std out every x games, also saves the network
        learn_rate (float):
        batch_size (int):

    Returns:
        (variables used in the final network : list, win rate: float)
    """

    p1wins = np.array([])
    p2wins = np.array([])
    drawsarr = np.array([])

    save_network_file_path = save_network_file_path or network_file_path

    input_layer, output_layer, variables = create_network()
    input_layer_2, output_layer_2, variables_2 = create_network_2()
    
    target_1 = tf.placeholder("float", shape=(None))
    target_2 = tf.placeholder("float", shape=(None))
    
    actual_move_placeholder = tf.placeholder("float", shape=(game_spec.outputs()))
    actual_move_placeholder_2 = tf.placeholder("float", shape=(game_spec.outputs()))
    
    prediction = tf.reduce_sum(actual_move_placeholder*output_layer)
    prediction_2 = tf.reduce_sum(actual_move_placeholder_2*output_layer_2)
    
    td_gradient_1 = tf.reduce_mean(tf.square(prediction - target_1))
    td_gradient_2 = tf.reduce_mean(tf.square(prediction_2 - target_2))
    
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(td_gradient_1)
    train_step_2 = tf.train.AdamOptimizer(learn_rate).minimize(td_gradient_2)

    Q_value_log = []

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        if network_file_path and os.path.isfile(network_file_path):
            print("loading pre-existing network")
            load_network(session, variables, network_file_path)

        mini_batch_board_states, mini_batch_moves, mini_batch_rewards = [], [], []
        results = collections.deque(maxlen=print_results_every)

        mini_batch_board_states_2, mini_batch_moves_2, mini_batch_rewards_2 = [], [], []
        results_2 = collections.deque(maxlen=print_results_every)

        def make_training_move(board_state, side, eps):
            mini_batch_board_states.append(np.ravel(board_state) * side)
            move = get_td_network_move(session, input_layer, output_layer, board_state, side, eps, valid_only=True, game_spec=game_spec) # valid_only=True, game_spec=game_spec
            mini_batch_moves.append(move)
            return game_spec.flat_move_to_tuple(move.argmax())
        
        
        
        def make_training_move_2(board_state, side, eps): 
            """
            To have the second player play randomly, change positional argument eps in get_td_network_move() to eps=1.
            To have the second player play epsilon greedily, leave positional argument eps in get_td_network_move() as eps from 
            make_training_move_2(..., eps).
            """
            mini_batch_board_states_2.append(np.ravel(board_state) * side)
            move = get_td_network_move(session, input_layer_2, output_layer_2, board_state, side, eps=1, valid_only=True, game_spec=game_spec) # valid_only=True, game_spec=game_spec
            mini_batch_moves_2.append(move)
            return game_spec.flat_move_to_tuple(move.argmax())
        
###############################   Training   ##################################
            
        for episode_number in range(1, number_of_games):
            

            log_ = False
            eps = 5000/episode_number
            #if episode_number%5000 == 0: # shows game after every 5000 episodes
            #    log_ =  True
            if (not randomize_first_player) or bool(random.getrandbits(1)):
                reward = game_spec.play_game_eps(make_training_move, make_training_move_2, eps, log = log_) # log = log_
                reward_2 = - reward
            else:
                reward = -game_spec.play_game_eps(make_training_move_2, make_training_move, eps, log = log_)
                reward_2 = - reward

            results.append(reward)
            results_2.append(reward_2)
            
            # we scale here so winning quickly is better winning slowly and loosing slowly better than loosing quick
            last_game_length = len(mini_batch_board_states) - len(mini_batch_rewards)
            last_game_length_2 = len(mini_batch_board_states_2) - len(mini_batch_rewards_2)

            reward /= float(last_game_length)
            reward_2 /= float(last_game_length_2)

            mini_batch_rewards += ([reward] * (last_game_length))# This applies a reward to the whole game!!
            mini_batch_rewards_2 += ([reward_2] * last_game_length_2) # Changes learning dynmics. No sparse reward environment anymore.


###################   applying TD learning after every game. -> unstable, Q-values will diverge
            
            for i in range(len(mini_batch_rewards)-1):
                state = mini_batch_board_states[i]
                state = state.reshape(1, *input_layer.get_shape().as_list()[1:])
                next_state = mini_batch_board_states[i+1]
                next_state = next_state.reshape(1, *input_layer.get_shape().as_list()[1:])
                move = mini_batch_moves[i]
                reward = mini_batch_rewards[i]
                Q_value_of_actions = session.run(output_layer,
                    feed_dict={input_layer: next_state})[0]
                max_Q_value = np.max(Q_value_of_actions)
                target = reward + 0.95*max_Q_value
         
                session.run(train_step, feed_dict={input_layer: state, actual_move_placeholder: move, target_1: target})
                
###############################################################################  
            
            #for i in range(len(mini_batch_rewards_2)-1):
            #    state = mini_batch_board_states_2[i]
            #    state = state.reshape(1, *input_layer_2.get_shape().as_list()[1:])
            #    next_state = mini_batch_board_states_2[i+1]
            #    next_state = next_state.reshape(1, *input_layer_2.get_shape().as_list()[1:])
            #    move = mini_batch_moves_2[i]
            #    reward = mini_batch_rewards_2[i]
            #    Q_value_of_actions = session.run(output_layer_2,
            #        feed_dict={input_layer_2: next_state})[0]
            #    max_Q_value = np.max(Q_value_of_actions)
            #    target = reward + 0.99*max_Q_value
            #    if episode_number % 1000 == 0:
            #        print(f'reward: {reward}')
            #        print(f'Q_values: {Q_value_of_actions}')
            #        print(f'move: {move}')
            #       print(f'target: {target}')
                #session.run(train_step_2, feed_dict={input_layer: state, target_1: target})
                # clear batches
            del mini_batch_board_states[:]
            del mini_batch_moves[:]
            del mini_batch_rewards[:]
           #     """"""
            del mini_batch_board_states_2[:]
            del mini_batch_moves_2[:]
            del mini_batch_rewards_2[:]
           #     """"""
           #just saving the Q-value for one state to show divergence of Q-Values
            Q_value_log += [list(session.run(output_layer,
                    feed_dict={input_layer: np.expand_dims([0,0,0,0,0,0,0,0,0],0)})[0])[0]]
    
    
###############################   Results    ##################################
    
            if episode_number % print_results_every == 0:
                draws = sum([x == 0 for x in results])
                print(" Player 1: episode: %s win_rate: %s" % (episode_number, _win_rate_strict(print_results_every, results)))
                print(" Player 2: episode: %s win_rate: %s" % (episode_number, _win_rate_strict(print_results_every, results_2)))
                print(f'Proportion of Draws: = {draws/print_results_every}')

                p1wins = np.append(p1wins, _win_rate_strict(print_results_every, results))
                p2wins = np.append(p2wins, _win_rate_strict(print_results_every, results_2))
                drawsarr = np.append(drawsarr, draws/print_results_every)
            
        if network_file_path:
            save_network(session, variables, save_network_file_path)
            save_network(session, variables_2, save_network_file_path)
            
###############################   Plotting   ##################################
            
    plt.plot(Q_value_log)
    plt.title("Q-value-Divergence - Deadly Triad")
    plt.xlabel("iterations")
    plt.ylabel("Q-values")
    #plt.savefig('Q-value-Divergence.png')
    plt.show()
    
###############################################################################
    
    return p1wins, p2wins, drawsarr
    #return variables, _win_rate(print_results_every, results)



def _win_rate(print_results_every, results):
    return 0.5 + sum(results) / (print_results_every * 2.)

def _win_rate_strict(print_results_every, results):
    wins = sum([x == 1 for x in results])
    return wins / (print_results_every)