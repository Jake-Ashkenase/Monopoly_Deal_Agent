import numpy as np
import copy
from Game_Representation_Minimax import *

class MonopolyDealMinimax:
    def __init__(self, agent_hand=[], agent_board_prop=np.zeros(10), agent_board_cash=np.zeros(6), opponent_hand=[], opponent_board_prop=np.zeros(10), opponent_board_cash=np.zeros(6), deck_quantities=np.array([]), depth=3):
        self.state = {
        'agent_hand': copy.copy(agent_hand),
        'agent_board_prop': np.copy(agent_board_prop),
        'agent_board_cash': copy.copy(agent_board_cash),
        'opponent_hand': copy.copy(opponent_hand),
        'opponent_board_prop': np.copy(opponent_board_prop),
        'opponent_board_cash': copy.copy(opponent_board_cash),
        'deck_quantities': np.copy(deck_quantities)
        }
        self.depth = depth

    def minimax(self, state, depth, alpha, beta, agent):
        game = MonopolyDealMinimaxEnv(state['agent_hand'], state['agent_board_prop'], state['agent_board_cash'], state['opponent_hand'], state['opponent_board_prop'], state['opponent_board_cash'], state['deck_quantities'])
        if depth == 0 or game.game_over(agent) or game.game_over(not agent) or game.draw():
            return game.evaluate(True), None
        
        # agent (True) is maximizing player
        if agent:
            max_eval = -np.inf
            best_move = None
            for move in state['agent_hand']:
                state_copy = MonopolyDealMinimaxEnv(copy.copy(state['agent_hand']), np.copy(state['agent_board_prop']), copy.copy(state['agent_board_cash']), copy.copy(state['opponent_hand']), np.copy(state['opponent_board_prop']), copy.copy(state['opponent_board_cash']), np.copy(state['deck_quantities']))
                new_state, reward, done = state_copy.step(move, agent)
                eval_score, _ = self.minimax(new_state, depth-1, alpha, beta, not agent)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = np.inf
            best_move = None
            for move in state['opponent_hand']:
                state_copy = MonopolyDealMinimaxEnv(copy.copy(state['agent_hand']), np.copy(state['agent_board_prop']), copy.copy(state['agent_board_cash']), copy.copy(state['opponent_hand']), np.copy(state['opponent_board_prop']), copy.copy(state['opponent_board_cash']), np.copy(state['deck_quantities']))
                new_state, reward, done = state_copy.step(move, agent)
                eval_score, _ = self.minimax(new_state, depth-1, alpha, beta, not agent)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def choose_best_action(self, agent):
        _, best_move = self.minimax(self.state, self.depth, -np.inf, np.inf, agent)
        return best_move
