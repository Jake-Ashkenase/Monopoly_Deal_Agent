from Game_Representation import *
import copy, math, random
import gym
from gym import spaces
import numpy as np

# Global win condition: complete this many property sets to win.
WIN_SET_COUNT = 3

"""
Limitations on this representation:
- No wild properties
- Only action card is rent
"""

class MonopolyDealEnv(gym.Env):
    def __init__(self):
        super(MonopolyDealEnv, self).__init__()

        # define the hand of the agent and opponent, a max of 7 cards 
        self.agent_hand = []
        self.opponent_hand = []

        # indexing colors for the property board representation
        self.color_to_index = {
            'P_Green': 1,
            'P_DBlue': 2,
            'P_Brown': 3,
            'P_LBlue': 4,
            'P_Orange': 5,
            'P_Pink': 6,
            'P_Black': 7,
            'P_Red': 8,
            'P_Tan': 9,
            'P_Yellow': 10
        }
        
        self.cash_to_index = {
            'One_Cash': 1,
            'Two_Cash': 2,
            'Three_Cash': 3,
            'Four_Cash': 4,
            'Five_Cash': 5,
            'Ten_Cash': 6,
        }

        # Complete set requirements for each property color.
        self.color_to_complete_set = np.array([3, 2, 2, 3, 3, 3, 4, 3, 2, 3])
        
        # Agent board: vector for each color (number of property cards played)
        self.agent_board_prop = np.zeros(10)  
        self.agent_board_cash = np.zeros(6)

        # Opponent board.
        self.opponent_board_prop = np.zeros(10)  
        self.opponent_board_cash = np.zeros(6)

        self.rewards = {
            'Goal': 100,  # bonus for winning by completing required sets
            'Cash': 0,  
            'Steal Cash': 0,  
            'Place Property': 0,  
            'Steal Property': 0,  
            'Set': 10
        }

        self.deck = {
            'One_Cash': {'action' : None, 'value' : 1, 'prop_color': None},
            'Two_Cash': {'action' : None, 'value' : 2, 'prop_color': None},
            'Three_Cash': {'action' : None, 'value' : 3, 'prop_color': None},
            'Four_Cash': {'action' : None, 'value' : 4, 'prop_color': None},
            'Five_Cash': {'action' : None, 'value' : 5, 'prop_color': None},
            'Ten_Cash': {'action' : None, 'value' : 10, 'prop_color': None},
            'P_Green': {'action' : None, 'value' : 4, 'prop_color': 'P_Green'},
            'P_DBlue': {'action' : None, 'value' : 4, 'prop_color': 'P_DBlue'},
            'P_Brown': {'action' : None, 'value' : 1, 'prop_color': 'P_Brown'},
            'P_LBlue': {'action' : None, 'value' : 1, 'prop_color': 'P_LBlue'},
            'P_Orange': {'action' : None, 'value' : 2, 'prop_color': 'P_Orange'},
            'P_Pink': {'action' : None, 'value' : 2, 'prop_color': 'P_Pink'},
            'P_Black': {'action' : None, 'value' : 2, 'prop_color': 'P_Black'},
            'P_Red': {'action' : None, 'value' : 3, 'prop_color': 'P_Red'},
            'P_Tan': {'action' : None, 'value' : 2, 'prop_color': 'P_Tan'},
            'P_Yellow': {'action' : None, 'value' : 3, 'prop_color': 'P_Yellow'},
            # 'House' and 'Hotel' cards are omitted for simplicity.
            'Rent': {'action' : 'rent', 'value' : 3, 'prop_color': 'Any'},
        }

        # Quantities for each card.
        self.deck_quantities = np.array([6, 5, 3, 3, 2, 1, 3, 2, 2, 3, 3, 3, 4, 3, 2, 3, 5])

        # Rent prices table.
        self.rent_prices = np.array([
            [0, 2, 4, 7, 7],  # P_Green
            [0, 3, 8, 8, 8],  # P_DBlue
            [0, 1, 2, 2, 2],  # P_Brown
            [0, 1, 2, 3, 3],  # P_LBlue
            [0, 1, 3, 5, 5],  # P_Orange
            [0, 1, 2, 4, 4],  # P_Pink
            [0, 1, 2, 3, 4],  # P_Black
            [0, 2, 3, 6, 6],  # P_Red
            [0, 1, 2, 2, 2],  # P_Tan
            [0, 2, 4, 6, 6]   # P_Yellow
        ])

        # Gym API spaces
        self.action_space = spaces.Discrete(len(self.deck))
        self.observation_space = spaces.Dict({
            'agent_hand': spaces.MultiBinary(len(self.deck)),
            'agent_board_prop': spaces.Box(0, np.inf, shape=(10,), dtype=int),
            'agent_board_cash': spaces.Box(0, np.inf, shape=(6,), dtype=int),
            'opponent_board_prop': spaces.Box(0, np.inf, shape=(10,), dtype=int),
            'opponent_board_cash': spaces.Box(0, np.inf, shape=(6,), dtype=int),
        })

    def reset(self):
        """Reset the game state."""
        self.agent_hand = []
        self.opponent_hand = []
        self.agent_board_prop = np.zeros(10)
        self.opponent_board_prop = np.zeros(10)
        self.agent_board_cash = np.zeros(6)
        self.opponent_board_cash = np.zeros(6)
        self.deck_quantities = np.array([6, 5, 3, 3, 2, 1, 3, 2, 2, 3, 3, 3, 4, 3, 2, 3, 5])
        for i in range(7):
            self.draw_card(True)
            self.draw_card(False)
        return self.get_observation()

    def get_observation(self):
        """Return current game state observation."""
        return {
            'agent_hand': self.agent_hand,
            'agent_board_prop': self.agent_board_prop,
            'opponent_board_prop': self.opponent_board_prop,
            'agent_board_cash': self.agent_board_cash,
            'opponent_board_cash': self.opponent_board_cash,
        }
    
    def num_completed_sets(self, agent):
        board_prop = self.agent_board_prop if agent else self.opponent_board_prop
        return sum(1 for a, b in zip(board_prop, self.color_to_complete_set) if a >= b)
    
    def game_over(self, agent):
        if self.num_completed_sets(agent) >= WIN_SET_COUNT:
            return (True, self.rewards['Goal'])
        return (False, 0)
                
    def calculate_rent(self, agent):
        rent_options = self.rent_prices[np.arange(10),
            self.agent_board_prop.astype(int) if not agent else self.opponent_board_prop.astype(int)]
        return np.max(rent_options)
    
    def play_rent_action(self, rent_value, agent):
        current_rent = 0
        card_values = np.array([val['value'] for _, val in self.deck.items()])
        card_values_weighted = np.array([
            val['value'] * 4 if card.startswith('P_') else val['value']
            for card, val in self.deck.items() if val['prop_color'] != 'Any'
        ])
        while current_rent < rent_value and len(self.agent_hand if not agent else self.opponent_hand) > 0:
            cash_and_props = np.append(
                np.where(self.opponent_board_cash != 0, 1, 0) if agent else np.where(self.agent_board_cash != 0, 1, 0),
                self.opponent_board_prop if agent else self.agent_board_prop
            )
            weights = np.multiply(cash_and_props, card_values_weighted)
            weights = np.where(weights == 0, np.inf, weights)
            min_index = np.argmin(weights)
            if min_index < len(weights) - len(self.color_to_index):
                if agent:
                    self.opponent_board_cash[min_index] -= 1
                    self.agent_board_cash[min_index] += 1
                else:
                    self.agent_board_cash[min_index] -= 1
                    self.opponent_board_cash[min_index] += 1
            else:
                idx = min_index - (len(weights) - len(self.color_to_index))
                if agent:
                    self.opponent_board_prop[idx] -= 1
                    self.agent_board_prop[idx] += 1
                else:
                    self.agent_board_prop[idx] -= 1
                    self.opponent_board_prop[idx] += 1
            current_rent += card_values[min_index]
        return current_rent
    
    def draw_card(self, agent):
        """Draw a card using weighted random sampling based on remaining quantities."""
        if np.sum(self.deck_quantities) > 0:
            cards = list(self.deck.keys())
            weights = self.deck_quantities / np.sum(self.deck_quantities)
            selected_idx = random.choices(range(len(cards)), weights=weights, k=1)[0]
            self.deck_quantities[selected_idx] -= 1
            card_drawn = cards[selected_idx]
            if agent:
                self.agent_hand.append(card_drawn)
            else:
                self.opponent_hand.append(card_drawn)
            return card_drawn

    def step(self, action, agent=True):
        done = False
        card = self.deck[action]
        reward = 0
        sets_before = self.num_completed_sets(agent)
        if agent:
            self.agent_hand.remove(action)
        else:
            self.opponent_hand.remove(action)
        if card['action']:
            rent_value = self.calculate_rent(agent)
            rent_received = self.play_rent_action(rent_value, agent)
            reward += rent_received
        elif card['prop_color']:
            idx = self.color_to_index[card['prop_color']] - 1
            if agent:
                self.agent_board_prop[idx] += 1
            else:
                self.opponent_board_prop[idx] += 1
            reward += (2 * card['value'])
        else:
            if agent:
                self.agent_board_cash[self.cash_to_index[action] - 1] += 1
            else:
                self.opponent_board_cash[self.cash_to_index[action] - 1] += 1
            reward += card['value']

        if self.num_completed_sets(agent) > sets_before:
            reward += 20

        game_finished, game_reward = self.game_over(agent)
        if game_finished:
            done = True
            reward += game_reward
        return self.get_observation(), reward if agent else -reward, done

    def select_random_action(self, agent):
        hand = self.agent_hand if agent else self.opponent_hand
        if not hand:
            return None
        return random.choice(hand)

    # --- HEURISTIC ROLLOUT FUNCTION ---
    def heuristic_rollout_move(self, agent_turn):
        hand = self.agent_hand if agent_turn else self.opponent_hand
        board = self.agent_board_prop if agent_turn else self.opponent_board_prop

        for move in hand:
            card = self.deck[move]
            if card['prop_color'] and card['prop_color'] in self.color_to_index:
                idx = self.color_to_index[card['prop_color']] - 1
                if board[idx] + 1 == self.color_to_complete_set[idx]:
                    return move
        return random.choice(hand) if hand else None

    def monte_carlo_tree_search(self, iterations=1000, agent=True, rollout_depth=10, c_param=1.414):
        class MCTSNode:
            def __init__(self, state, parent, move, agent):
                self.state = state
                self.parent = parent
                self.move = move
                self.agent = agent
                self.children = []
                self.wins = 0
                self.visits = 0
                self.untried_moves = list(state.agent_hand) if agent else list(state.opponent_hand)
                self.reward = 0

            def is_terminal(self):
                done, _ = self.state.game_over(self.agent)
                return done

            def is_fully_expanded(self):
                return len(self.untried_moves) == 0

            def expand(self):
                move = self.untried_moves.pop()
                new_state = copy.deepcopy(self.state)
                _, immediate_reward, _ = new_state.step(move, self.agent)
                child_node = MCTSNode(new_state, parent=self, move=move, agent=not self.agent)
                child_node.reward = immediate_reward
                self.children.append(child_node)
                return child_node

            def best_child(self, c_param):
                best_score = -float('inf')
                best_child = None
                for child in self.children:
                    if child.visits == 0:
                        score = float('inf')
                    else:
                        exploit = child.wins / child.visits
                        explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
                        score = exploit + explore
                    if score > best_score:
                        best_score, best_child = score, child
                return best_child

            def update(self, result):
                self.visits += 1
                self.wins += result

        def rollout(state, current_agent, rollout_depth):
            sim_state = copy.deepcopy(state)
            agent_turn = current_agent
            depth = 0
            cum_reward = 0
            while depth < rollout_depth:
                done, reward = sim_state.game_over(agent_turn)
                if done:
                    return reward
                moves = sim_state.agent_hand if agent_turn else sim_state.opponent_hand
                if not moves:
                    break
                move = sim_state.heuristic_rollout_move(agent_turn)
                if move is None:
                    break
                _, r, done = sim_state.step(move, agent_turn)
                cum_reward += r
                if done:
                    return cum_reward
                agent_turn = not agent_turn
                depth += 1
            return cum_reward

        root = MCTSNode(copy.deepcopy(self), parent=None, move=None, agent=agent)
        for _ in range(iterations):
            node = root
            # SELECTION: descend until terminal or a node that still has untried moves
            while not node.is_terminal() and node.is_fully_expanded():
                if not node.children:
                    # no moves to pick, treat as terminal for MCTS purposes
                    break
                node = node.best_child(c_param)

            # EXPANSION: only if it isn’t terminal and there are untried moves
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # SIMULATION
            result = node.reward + rollout(node.state, node.agent, rollout_depth)

            # BACKPROPAGATION
            while node is not None:
                node.update(result)
                result = -result
                node = node.parent

        if root.children:
            return max(root.children, key=lambda c: c.visits).move
        return None

# --- RUN SIMULATION AND TESTS ---

def run_simulation(num_games=1000, base_iterations=200, max_turns=150, rollout_depth=None):
    agent_wins = opponent_wins = draws = 0
    total_sim_reward = 0

    for game in range(num_games):
        env = MonopolyDealEnv()
        env.reset()
        initial_deck = np.sum(env.deck_quantities)
        done = False
        turn = 0
        game_reward = 0

        while not done and turn < max_turns:
            remaining = np.sum(env.deck_quantities)
            frac = remaining / initial_deck
            its = max(50, int(base_iterations * frac))
            # use fixed rollout depth if provided, else dynamic
            rd = rollout_depth if rollout_depth is not None else (1 + turn // (max_turns // 3))
            c = 1.0 + (turn / max_turns)

            if turn % 2 == 0:
                move = env.monte_carlo_tree_search(
                    iterations=its,
                    agent=True,
                    rollout_depth=rd,
                    c_param=c
                )
                if move is None:
                    break
                _, r, done = env.step(move, agent=True)
            else:
                move = env.select_random_action(agent=False)
                if move is None:
                    break
                _, r, done = env.step(move, agent=False)

            game_reward += r
            while len(env.agent_hand if turn%2==0 else env.opponent_hand) < 5 and np.sum(env.deck_quantities) > 0:
                env.draw_card(turn%2==0)

            turn += 1

        total_sim_reward += game_reward
        # Determine winner
        if env.num_completed_sets(True) >= WIN_SET_COUNT:
            agent_wins += 1
        elif env.num_completed_sets(False) >= WIN_SET_COUNT:
            opponent_wins += 1
        else:
            a_sets = env.num_completed_sets(True)
            o_sets = env.num_completed_sets(False)
            if a_sets > o_sets:
                agent_wins += 1
            elif o_sets > a_sets:
                opponent_wins += 1
            else:
                draws += 1

        if (game + 1) % 100 == 0:
            print(f"Completed {game + 1} games...")

    print(f"\n--- Simulation Results (rollout depth={rollout_depth if rollout_depth is not None else 'dynamic'}) ---")
    print(f"Total games played: {num_games}")
    print(f"Agent wins: {agent_wins} ({agent_wins/num_games*100:.2f}%)")
    print(f"Opponent wins: {opponent_wins} ({opponent_wins/num_games*100:.2f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.2f}%)")
    print(f"Total simulation reward: {total_sim_reward}")

# --- TESTS ---

def test_heuristic_rollout():
    """
    Test that the heuristic rollout move chooses a property card that would complete a set.
    For this test, we set the agent's board such that one property is one short of completion,
    and include that property in the hand.
    """
    env = MonopolyDealEnv()
    env.reset()
    # For P_DBlue, complete set requires 2. Set board to 1.
    idx = env.color_to_index['P_DBlue'] - 1
    env.agent_board_prop[idx] = 1
    # Ensure hand contains the property card.
    env.agent_hand = ['P_DBlue', 'Ten_Cash']
    move = env.heuristic_rollout_move(True)
    print("Test Heuristic Rollout: Selected move:", move)
    assert move == 'P_DBlue', f"Expected 'P_DBlue' but got {move}"

def test_game_over():
    """
    Test the game_over function by artificially setting the board such that
    the agent has the required complete sets.
    """
    env = MonopolyDealEnv()
    env.reset()
    # Set three colors to be complete.
    env.agent_board_prop[0] = 3   # P_Green
    env.agent_board_prop[1] = 2   # P_DBlue (requirement 2)
    env.agent_board_prop[2] = 2   # P_Brown (requirement 2)
    done, reward = env.game_over(True)
    print("Test Game Over: done =", done, "reward =", reward)
    assert done == True and reward == env.rewards['Goal'], "Game should be over with goal reward."

def test_draw_card():
    """
    Test that drawing a card adds to the hand and reduces the deck quantity.
    """
    env = MonopolyDealEnv()
    env.reset()
    initial_deck_sum = np.sum(env.deck_quantities)
    initial_hand_len = len(env.agent_hand)
    drawn = env.draw_card(True)
    new_deck_sum = np.sum(env.deck_quantities)
    new_hand_len = len(env.agent_hand)
    print("Test Draw Card: Drawn card =", drawn)
    assert new_hand_len == initial_hand_len + 1, "Hand should increase by one."
    assert new_deck_sum == initial_deck_sum - 1, "Deck quantities should decrease by one."

def test_step_cash_card():
    """
    Test that playing a cash card updates the board cash correctly.
    """
    env = MonopolyDealEnv()
    env.reset()
    env.agent_hand = ['Ten_Cash']
    initial_cash = env.agent_board_cash.copy()
    _, reward, done = env.step('Ten_Cash', agent=True)
    print("Test Step Cash Card: Reward =", reward, "done =", done)
    expected_cash = initial_cash.copy()
    expected_cash[env.cash_to_index['Ten_Cash'] - 1] += 1
    assert np.array_equal(env.agent_board_cash, expected_cash), "Agent cash board not updated correctly."

def test_step_property_card():
    """
    Test that playing a property card updates the board property correctly.
    """
    env = MonopolyDealEnv()
    env.reset()
    env.agent_hand = ['P_Brown']
    idx = env.color_to_index['P_Brown'] - 1
    initial_count = env.agent_board_prop[idx]
    _, reward, done = env.step('P_Brown', agent=True)
    print("Test Step Property Card: Reward =", reward, "done =", done)
    assert env.agent_board_prop[idx] == initial_count + 1, "Agent property board not updated correctly."

def test_rent_action():
    """
    Test that playing a rent card returns a positive reward if the opponent has cards.
    """
    env = MonopolyDealEnv()
    env.reset()
    env.opponent_hand = ['Two_Cash']
    idx = env.color_to_index['P_DBlue'] - 1
    env.opponent_board_prop[idx] = 2
    env.agent_hand = ['Rent']
    _, reward, done = env.step('Rent', agent=True)
    print("Test Rent Action: Reward =", reward, "done =", done)
    assert reward > 0, "Rent action should yield a positive reward when opponent has cards."

def test_mcts_property_completion():
    env = MonopolyDealEnv()
    env.reset()
    idx = env.color_to_index['P_DBlue'] - 1
    env.agent_board_prop[idx] = 1
    env.agent_hand = ['P_DBlue', 'Ten_Cash']
    best_move = env.monte_carlo_tree_search(iterations=500, agent=True, rollout_depth=1)
    print("Test 1 - Property Completion: Best move selected:", best_move)
    assert best_move == 'P_DBlue', f"Expected 'P_DBlue' but got {best_move}"

def test_mcts_cash_card():
    env = MonopolyDealEnv()
    env.reset()
    env.agent_board_prop = np.zeros(10)
    env.agent_hand = ['P_Brown', 'Ten_Cash']
    best_move = env.monte_carlo_tree_search(iterations=500, agent=True)
    print("Test 2 - Cash Card: Best move selected:", best_move)
    assert best_move == 'Ten_Cash', f"Expected 'Ten_Cash' but got {best_move}"

def test_mcts_single_move():
    env = MonopolyDealEnv()
    env.reset()
    env.agent_hand = ['Two_Cash']
    best_move = env.monte_carlo_tree_search(iterations=100, agent=True)
    print("Test 3 - Single Move: Best move selected:", best_move)
    assert best_move == 'Two_Cash', f"Expected 'Two_Cash' but got {best_move}"

def run_all_tests():
    test_heuristic_rollout()
    test_game_over()
    test_draw_card()
    test_step_cash_card()
    test_step_property_card()
    test_rent_action()
    test_mcts_property_completion()
    test_mcts_cash_card()
    test_mcts_single_move()
    print("All tests passed!")
  
if __name__ == "__main__":
    run_all_tests()
    print()
    # Loop simulations over different rollout depths
    for rd in [1, 2, 5, 10]:
        print(f"\n=== Running simulation with rollout depth {rd} ===")
        run_simulation(num_games=1000, base_iterations=150, max_turns=150, rollout_depth=rd)
    print("\n=== Running simulation with dynamic rollout depth ===")
    run_simulation(num_games=1000, base_iterations=150, max_turns=150, rollout_depth=None)