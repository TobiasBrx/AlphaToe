"""
Created on Tue Apr 23 2019

@author: tobiasbraun

It is important to note that if the second player is randomized without knowing the rules,
their performance is so bad that the DQN algorithm simply learns how to play legal moves
and waits until the random opponent makes an illegal move which causes the DQN agent to win.
It is extremely unlikely that a randomized opponent wins or even draws under these curcumstances.
We achieve a win rate of >99% for the DQN agent against such a randomized opponent with around 0.2% 
winning games for the randomized opponent and 0.4 drawing games.
"""
"""
On the other hand when we allow the randomized opponent to know the rules of the game we can
witness a beautiful learning curve of the DQN agent from not knowing the rules to 
fully understanding the game and playing on a high (optimal?) level.
Starting from <5% winning games in the first 1000 games to about 50% after 50000 games (that corresponds
to playing as good as a randomized opponent who knows the rules and can thus be interpreted as
having learned the rules).
After about 100000 episodes the DQN agent dominates the random opponent with about 80% winning chances
but he does not find the Nash equilibrium strategy to not lose any games. Training from then on 
accomplishes an increase in performance to about 98% winning rate but again the rest of the games 
are lost and not drawn.
It is also interesting to note that the agent prioritizes to prevent the opponent from winning over 
winning themselve. If it is their turn and to win in 1 move they prefer to stop the opponent from winning
in 1 move and play on.

"""
###############################   Imports    ##################################

import collections
import os
import random
import numpy as np
import tensorflow as tf
from common.network_helpers import load_network, save_network

###############################################################################

def get_td_network_move(session, input_layer, output_layer, board_state, side, eps=0.1,
                                valid_only=False, game_spec=None, ):
    """Choose a move for the given board_state using a stocastic policy. A move is selected using epsilon greedy
    strategy of the values from the output_layer. With epsilon probability a random move is chosen and with
    1-epsilon probability the move with the highest Q-value is chosen.
    
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
            return move
        available_moves_flat = [game_spec.tuple_move_to_flat(x) for x in available_moves]
        
        if np.random.rand() < eps:
            pick = random.choice(available_moves_flat)
            move = np.zeros(game_spec.board_squares())
            np.put(move, pick, 1)
            return move
        for i in range(game_spec.board_squares()):
            if i not in available_moves_flat:
                Q_values_of_actions[i] = - np.inf
            
        pick = np.argmax(Q_values_of_actions)
        best_move = np.zeros(game_spec.board_squares())
        np.put(best_move, pick, 1)
        return best_move
        
    else:
        if np.random.rand() < eps:
            pick = random.choice(np.arange(9))
            move = np.zeros(game_spec.board_squares())
            np.put(move, pick, 1)
            return move
        else:
            pick = np.argmax(Q_values_of_actions)
            best_move = np.zeros(game_spec.board_squares())
            np.put(best_move, pick, 1)
    return best_move

###############################################################################

log_ = False # just for logging purposes

def DQN_train(game_spec,
                           create_network, # this should have scope principal
                           create_network_2,
                           create_target_network, # this should have scope "target
                           create_target_network_2,
                           network_file_path,
                           save_network_file_path=None,
                           opponent_func=None,
                           number_of_games=10000,
                           print_results_every=1000,
                           learn_rate=1e-4,
                           batch_size=100,
                           randomize_first_player=True):
    """Train a network using the DQN algorithm with replay buffer, a principal and a target network

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
    save_network_file_path = save_network_file_path or network_file_path

    input_layer, output_layer, variables = create_network()
    input_layer_2, output_layer_2, variables_2 = create_network_2()
    input_layer_t, output_layer_t, variables_t = create_target_network()
    input_layer_t2, output_layer_t2, variables_t2 = create_target_network_2()
    
    target_1 = tf.placeholder("float", shape=(None))
    target_2 = tf.placeholder("float", shape=(None))
    
    actual_move_placeholder = tf.placeholder("float", shape=(None, game_spec.outputs()))
    actual_move_placeholder_2 = tf.placeholder("float", shape=(None, game_spec.outputs()))
    
    prediction = tf.reduce_sum(actual_move_placeholder*output_layer, axis=1)
    prediction_2 = tf.reduce_sum(actual_move_placeholder_2*output_layer_2, axis=1)
    
    td_gradient_1 = tf.reduce_mean(tf.square(prediction - target_1))
    td_gradient_2 = tf.reduce_mean(tf.square(prediction_2 - target_2))
    
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(td_gradient_1)
    train_step_2 = tf.train.AdamOptimizer(learn_rate).minimize(td_gradient_2)
    
    gamma = 0.99
    tau = 100
###############################################################################    

#To copy the principal to the target network
    
    def build_target_update(from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_scope)
        op = []
        for v1, v2 in zip(from_vars, to_vars):
            op.append(v2.assign(v1))
        return op  
    
    update = build_target_update("principal", "target")

###############################################################################
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        if network_file_path and os.path.isfile(network_file_path):
            print("loading pre-existing network")
            load_network(session, variables, network_file_path)

        mini_batch_board_states, mini_batch_moves, mini_batch_rewards = [], [], []
        mini_batch_board_states_2, mini_batch_moves_2, mini_batch_rewards_2 = [], [], []
        mini_batch_board_states_temp, mini_batch_next_board_states = [], []
        mini_batch_board_states_temp_2, mini_batch_next_board_states_2 = [], []
        results = collections.deque(maxlen=print_results_every)
        results_2 = collections.deque(maxlen=print_results_every)
      
###############################################################################
        
        def make_training_move(board_state, side, eps):
            mini_batch_board_states.append(np.ravel(board_state) * side)
            #epsilon greedy choice of the next move
            move = get_td_network_move(session, input_layer, output_layer, board_state, 
                                       side, eps, valid_only=False, game_spec=game_spec) # valid_only=True, game_spec=game_spec
            mini_batch_moves.append(move)
            return game_spec.flat_move_to_tuple(move.argmax())
        
###############################################################################        

#It has the option of letting the user play the game when log_ is set to
# "Interactive"
            
        def make_training_move_2(board_state, side, eps):
            """
            To have the second player play randomly, change positional argument eps in get_td_network_move() to > eps=1 <.
            To have the second player play epsilon greedily, leave positional argument eps in get_td_network_move() as > eps < (from 
            make_training_move_2(..., eps)).
            """
            global log_
            mini_batch_board_states_2.append(np.ravel(board_state) * side)
            if log_ == "Interactive":
                pick = np.int(input("Enter Move (0-9): "))
                move = np.zeros(game_spec.board_squares())
                np.put(move, pick, 1)
                mini_batch_moves_2.append(move)
                return game_spec.flat_move_to_tuple((move.argmax()))
            #epsilon greedy choice of the next move
            else: move = get_td_network_move(session, input_layer_2, output_layer_2, board_state, 
                                       side, eps=1, valid_only=True, game_spec=game_spec) # valid_only=True
            mini_batch_moves_2.append(move)
            return game_spec.flat_move_to_tuple(move.argmax())

###############################   Training   ##################################
            
        for episode_number in range(1, number_of_games):
            global log_
            log_ = False
            if episode_number % 20000 == 0:
                log_=True
            elif (episode_number%50000) == 0 or (episode_number>400000 and episode_number %5000):
                log_ = "Interactive"
            eps = np.exp(-10*episode_number/number_of_games) #5000/episode_number # np.exp(-5*episode_number/200000)
            
            if (not randomize_first_player) or bool(random.getrandbits(1)):
                reward = game_spec.play_game_eps(make_training_move, make_training_move_2, eps, log = log_)
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
            
            mini_batch_rewards += ([reward] * (last_game_length))# remember that this applies a reward to the whole game!!
            #mini_batch_rewards += [0]*(last_game_length-1)+[reward]
            mini_batch_rewards_2 += ([reward_2] * last_game_length_2) # Changes learning dynmics. No sparse reward environment anymore.
            #mini_batch_rewards_2 += [0]*(last_game_length_2-1)+[reward_2]
            
            length = len(mini_batch_board_states_temp)
            mini_batch_board_states_temp = np.copy(mini_batch_board_states)
            new_moves = mini_batch_board_states[length:]
            mini_batch_next_board_states += (new_moves[1:]+[np.array([0,0,0,0,0,0,0,0,0])])

            length_2 = len(mini_batch_board_states_temp_2)
            mini_batch_board_states_temp_2 = np.copy(mini_batch_board_states_2)
            new_moves_2 = mini_batch_board_states_2[length_2:]
            mini_batch_next_board_states_2 += (new_moves_2[1:]+[np.array([0,0,0,0,0,0,0,0,0])])
                
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
                np_mini_batch_next_board_states = np.array(mini_batch_next_board_states) \
                    .reshape(len(mini_batch_rewards), *input_layer.get_shape().as_list()[1:])
                np_mini_batch_next_board_states_2 = np.array(mini_batch_next_board_states_2) \
                    .reshape(len(mini_batch_rewards_2), *input_layer_2.get_shape().as_list()[1:])

                Q_targets = np.max(session.run(output_layer_t,
                    feed_dict={input_layer_t: np_mini_batch_next_board_states}), axis=1)
                done = [0 if all([x == 0 for x in i]) else 1 for i in np_mini_batch_next_board_states]
                targets_ = mini_batch_rewards + gamma*Q_targets*done
                
                session.run(train_step, feed_dict={input_layer: np_mini_batch_board_states, \
                                                   actual_move_placeholder: mini_batch_moves, target_1: targets_})

                if (episode_number%tau == 0):
                    session.run(update)
                    
###############################################################################
                    
                # clear batches
                
                del mini_batch_board_states[:]
                del mini_batch_moves[:]
                del mini_batch_rewards[:]
                del mini_batch_board_states_2[:]
                del mini_batch_moves_2[:]
                del mini_batch_rewards_2[:]
                mini_batch_next_board_states = []
                mini_batch_next_board_states_2 = []
                length, length_2 = 0, 0
                mini_batch_board_states_temp = []
                mini_batch_board_states_temp_2 = []
                new_moves = []
                new_moves_2 = []
                
###############################   Results    ##################################

            if episode_number % print_results_every == 0:
                draws = sum([x == 0 for x in results])
                print(" Player 1: episode: %s win_rate: %s" % (episode_number, _win_rate_strict(print_results_every, results)))
                print(" Player 2: episode: %s win_rate: %s" % (episode_number, _win_rate_strict(print_results_every, results_2)))
                print(f'Proportion of Draws: = {draws/print_results_every}')
                if network_file_path:
                    save_network(session, variables, save_network_file_path)
                    save_network(session, variables_2, save_network_file_path)
            
            
            
            
####################     ANALYSIS & LOGGING ###################################
                    
            if episode_number % 50000 == 0:
                Q = session.run(output_layer_t,
                    feed_dict={input_layer_t: np.expand_dims([0,1,-1,-1,1,0,0,0,0],0)})
                print(f'Q-values: {Q}')
                Q = session.run(output_layer_t,
                    feed_dict={input_layer_t: np.expand_dims([0,1,1,-1,-1,0,0,0,0],0)})
                print(f'Q-values: {Q}')
                Q = session.run(output_layer_t,
                    feed_dict={input_layer_t: np.expand_dims([-1,-1,0,0,-1,1,1,1,0],0)})
                print(f'Q-values: {Q}')
            
###############################################################################
        
        if network_file_path:
            save_network(session, variables, save_network_file_path)
            save_network(session, variables_2, save_network_file_path)

    return variables, _win_rate(print_results_every, results)

###############################################################################

def _win_rate(print_results_every, results):
    return 0.5 + sum(results) / (print_results_every * 2.)

def _win_rate_strict(print_results_every, results):
    wins = sum([x == 1 for x in results])
    return wins / (print_results_every)
