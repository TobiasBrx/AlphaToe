from techniques.min_max import min_max_alpha_beta
from techniques.train_policy_gradient import train_policy_gradients
from tic_tac_toe_5_4.network import tic_tac_toe_5_4_game_spec, create_convolutional_network


def min_max_move_func(board_state, side):
    return min_max_alpha_beta(tic_tac_toe_5_4_game_spec, board_state, side, 3)[1]


train_policy_gradients(tic_tac_toe_5_4_game_spec, create_convolutional_network,
                       'pickles/DQN_Nash_well_trained_Network_Pickle',
                       opponent_func=min_max_move_func,
                       save_network_file_path='pickles/DQN_Nash_well_trained_Minmax_Network_Pickle',
                       number_of_games=50000,
                       print_results_every=1000)