import collections
import os
import random

import numpy as np
import tensorflow as tf

from common.network_helpers import load_network, get_stochastic_network_move, save_network
"""
This will have two agents have compete against each other with independent neural networks. (Vanilla version)
It turns out that this is extremely unstable.

"""
log_ = False
def train_policy_gradients(game_spec,
                           create_network,
                           create_network_2,
                           network_file_path,
                           save_network_file_path=None,
                           opponent_func=None,
                           number_of_games=10000,
                           print_results_every=1000,
                           learn_rate=1e-4,
                           batch_size=100,
                           randomize_first_player=True,
                           copy_network_at=0.65):
    """Train a network using policy gradients

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

    save_network_file_path = save_network_file_path #or network_file_path
    #opponent_func = opponent_func or game_spec.get_random_player_func()
    reward_placeholder = tf.placeholder("float", shape=(None,))
    actual_move_placeholder = tf.placeholder("float", shape=(None, game_spec.outputs()))

    reward_placeholder_2 = tf.placeholder("float", shape=(None,))
    actual_move_placeholder_2 = tf.placeholder("float", shape=(None, game_spec.outputs()))

    input_layer, output_layer, variables = create_network()
    input_layer_2, output_layer_2, variables_2 = create_network_2()

    policy_gradient = tf.log(
        tf.reduce_sum(tf.multiply(actual_move_placeholder, output_layer), reduction_indices=1)) * reward_placeholder
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(-policy_gradient)

    policy_gradient_2 = tf.log(
        tf.reduce_sum(tf.multiply(actual_move_placeholder_2, output_layer_2), reduction_indices=1)) * reward_placeholder_2
    train_step_2 = tf.train.AdamOptimizer(learn_rate).minimize(-policy_gradient_2)
    ###############################################################################    

#To copy the network from one player to the other 
    
    def build_target_update(from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_scope)
        op = []
        for v1, v2 in zip(from_vars, to_vars):
            op.append(v2.assign(v1))
        return op  
    

###############################################################################

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        if network_file_path and os.path.isfile(network_file_path):
            print(f"Loading pre-existing network from {network_file_path}")
            load_network(session, variables, network_file_path)

        mini_batch_board_states, mini_batch_moves, mini_batch_rewards = [], [], []
        results = collections.deque(maxlen=print_results_every)

        mini_batch_board_states_2, mini_batch_moves_2, mini_batch_rewards_2 = [], [], []
        results_2 = collections.deque(maxlen=print_results_every)

        def make_training_move(board_state, side):
            mini_batch_board_states.append(np.ravel(board_state) * side)
            
            
            move = get_stochastic_network_move(session, input_layer, output_layer, board_state, side, valid_only=False, game_spec=game_spec) # valid_only=True, game_spec=game_spec
            #print(f'move: {move}')
            mini_batch_moves.append(move)
            return game_spec.flat_move_to_tuple(move.argmax())
        
        def make_training_move_2(board_state, side):
            global log_
            mini_batch_board_states_2.append(np.ravel(board_state) * side)
            if log_ == "Interactive":
                pick = np.int(input("Enter Move (0-9): "))
                move = np.zeros(game_spec.board_squares())
                np.put(move, pick, 1)
                mini_batch_moves_2.append(move)
                return game_spec.flat_move_to_tuple((move.argmax()))
            else:
                move = get_stochastic_network_move(session, input_layer_2, output_layer_2, board_state, side, valid_only=False, game_spec=game_spec) # valid_only=True, game_spec=game_spec
                mini_batch_moves_2.append(move)
            return game_spec.flat_move_to_tuple(move.argmax())

        for episode_number in range(1, number_of_games):
            global log_ 
            log_ = False
            # randomize if going first or second
            if episode_number % 20000 == 0:
                log_=True
           # elif (episode_number%100001) == 0 or  (episode_number >499995 and episode_number<500000) or (
           #     episode_number >699995 and episode_number<700000) or (episode_number >799995 and episode_number<800000):
           #     log_ = "Interactive"
               
            if (not randomize_first_player) or bool(random.getrandbits(1)):
                reward = game_spec.play_game(make_training_move, make_training_move_2, log = log_) # log = log_
                reward_2 = - reward
            else:
                reward = -game_spec.play_game(make_training_move_2, make_training_move, log = log_)
                reward_2 = - reward

            results.append(reward)
            results_2.append(reward_2)
            
            # we scale here so winning quickly is better winning slowly and loosing slowly better than loosing quick
            last_game_length = len(mini_batch_board_states) - len(mini_batch_rewards)
            last_game_length_2 = len(mini_batch_board_states_2) - len(mini_batch_rewards_2)

            reward /= float(last_game_length)
            reward_2 /= float(last_game_length_2)

            mini_batch_rewards += ([reward] * last_game_length)
            mini_batch_rewards_2 += ([reward_2] * last_game_length_2)

            if episode_number % batch_size == 0:
                normalized_rewards = mini_batch_rewards - np.mean(mini_batch_rewards)
                normalized_rewards_2 = mini_batch_rewards_2 - np.mean(mini_batch_rewards_2)

                rewards_std = np.std(normalized_rewards)
                rewards_std_2 = np.std(normalized_rewards_2)
                    
                if rewards_std != 0:
                    normalized_rewards /= rewards_std
    
                else:
                    print("warning: got mini batch std of 0.")

                if rewards_std_2 != 0:
                    normalized_rewards_2 /= rewards_std_2

                else:
                    print("warning: got mini batch 2 std of 0.")
                    
                np_mini_batch_board_states = np.array(mini_batch_board_states) \
                    .reshape(len(mini_batch_rewards), *input_layer.get_shape().as_list()[1:])
                np_mini_batch_board_states_2 = np.array(mini_batch_board_states_2) \
                    .reshape(len(mini_batch_rewards_2), *input_layer_2.get_shape().as_list()[1:])

                session.run(train_step, feed_dict={input_layer: np_mini_batch_board_states,
                                                   reward_placeholder: normalized_rewards,
                                                   actual_move_placeholder: mini_batch_moves})
                """"""
                
                session.run(train_step_2, feed_dict={input_layer_2: np_mini_batch_board_states_2,
                                                   reward_placeholder_2: normalized_rewards_2,
                                                   actual_move_placeholder_2: mini_batch_moves_2})
                
          
                """"""
                # clear batches
                del mini_batch_board_states[:]
                del mini_batch_moves[:]
                del mini_batch_rewards[:]
                """"""
                del mini_batch_board_states_2[:]
                del mini_batch_moves_2[:]
                del mini_batch_rewards_2[:]
                """"""
            
            if episode_number % print_results_every == 0:
                draws = sum([x == 0 for x in results])
                print(" Player 1: episode: %s win_rate: %s" % (episode_number, _win_rate_strict(print_results_every, results)))
                print(" Player 2: episode: %s win_rate: %s" % (episode_number, _win_rate_strict(print_results_every, results_2)))
                print(f'Proportion of Draws: = {draws/print_results_every}')
                if network_file_path:
                    save_network(session, variables, save_network_file_path)
                    save_network(session, variables_2, save_network_file_path)
                
                p1wins = np.append(p1wins, _win_rate_strict(print_results_every, results))
                p2wins = np.append(p2wins, _win_rate_strict(print_results_every, results_2))
                drawsarr = np.append(drawsarr, draws/print_results_every)
            #   
           #     if (_win_rate_strict(print_results_every, results) > _win_rate_strict(print_results_every, results_2)+0.15):
           #         print("_______________________COPYING NETWORK_________________________")
           #         print("_____________________Player 1 -> Player 2______________________")
           #         update_winner_to_loser = build_target_update("player1", "player2")
           #         session.run(update_winner_to_loser)
                    
           #     elif (_win_rate_strict(print_results_every, results_2) > _win_rate_strict(print_results_every, results) +0.15):
           #         print("_______________________COPYING NETWORK_________________________")
           #         print("_____________________Player 2 -> Player 1______________________")
           #         
           #         update_winner_to_loser = build_target_update("player2", "player1")
           #         session.run(update_winner_to_loser)


        if save_network_file_path:
            print(f"Saving Network at {save_network_file_path}")
            save_network(session, variables, save_network_file_path)
            #save_network(session, variables_2, save_network_file_path)

    return p1wins, p2wins, drawsarr
    #return variables, _win_rate(print_results_every, results)



def _win_rate(print_results_every, results):
    return 0.5 + sum(results) / (print_results_every * 2.)

def _win_rate_strict(print_results_every, results):
    wins = sum([x == 1 for x in results])
    return wins / (print_results_every)
