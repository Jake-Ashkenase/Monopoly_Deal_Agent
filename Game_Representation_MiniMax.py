import gym 
import numpy as np 
import random

'''
Limitations on this representation:
- No wild properties
- Only action card is rent
'''


class MonopolyDealMinimaxEnv(gym.Env):
    def __init__(self, agent_hand=[], agent_board_prop=np.zeros(10), agent_board_cash=np.zeros(6), opponent_hand=[], opponent_board_prop=np.zeros(10), opponent_board_cash=np.zeros(6), deck_quantities=[]):
        super(MonopolyDealMinimaxEnv, self).__init__()

        # define the hand of the agent and opponent, a max of 7 cards 
        self.agent_hand = agent_hand
        self.opponent_hand = opponent_hand

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

        self.color_to_complete_set = np.array([3, 2, 2, 3, 3, 3, 4, 3, 2, 3])
        
        # define the board of the agent, with a vector for each color representing...
        #  [number of properties played, number of houses, number of hotels]
        self.agent_board_prop = agent_board_prop  # 10 colors
        self.agent_board_cash = agent_board_cash

        # define the board of the opponent, with a vector for each color representing...
        #  [number of properties played, number of houses, number of hotels]
        self.opponent_board_prop = opponent_board_prop  # 10 colors
        self.opponent_board_cash = opponent_board_cash

        self.rewards = {
            'Goal': 100,  # complete 3 sets 
            'Cash': 0,  # needs to be the value of cash placed 
            'Steal Cash': 0*2,  # needs to be the value of cash stolen, times 2
            'Place Property': 0,  # needs to be the value of the property placed
            'Steal Property': 0*2,  # needs to be the value of property stolen, times 2
            'Set': 10
        }

        self.deck= {
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
            # 'House': {'action' : None, 'value' : 3, 'prop_color': 'Set'},
            # 'Hotel': {'action' : None, 'value' : 4, 'prop_color': 'Set'},

            # leaving out wild properties for the simplest representation of the game

            'Rent': {'action' : 'rent', 'value' : 3, 'prop_color': 'Any'},
        }

        # define the quantities of each card in the deck

        self.deck_quantities = deck_quantities

        # rent reward is 0 if there are no properties played
        # Once the max number of properties is reached for a color, the rent remains the same as more properties are added
        self.rent_prices = np.array([
            [0, 2, 4, 7, 7], # P_Green
            [0, 3, 8, 8, 8], # P_DBlue
            [0, 1, 2, 2, 2], # P_Brown
            [0, 1, 2, 3, 3], # P_LBlue
            [0, 1, 3, 5, 5], # P_Orange
            [0, 1, 2, 4, 4], # P_Pink
            [0, 1, 2, 3, 4], # P_Black
            [0, 2, 3, 6, 6], # P_Red
            [0, 1, 2, 2, 2], # P_Tan
            [0, 2, 4, 6, 6]  # P_Yellow
        ])

        # Define action space s
        # self.actions = ['CARD1', 'CARD2', 'CARD3', 'CARD4', 'CARD5', 'CARD6', 'CARD7', 'PASS']


    def reset(self):
        '''
        Reset the game
        '''
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
        '''
        Get the observation of the game
        '''
        return {
            'agent_hand': self.agent_hand,
            'opponent_hand': self.opponent_hand,
            'agent_board_prop': self.agent_board_prop,
            'opponent_board_prop': self.opponent_board_prop,
            'agent_board_cash': self.agent_board_cash,
            'opponent_board_cash': self.opponent_board_cash,
            'deck_quantities': self.deck_quantities
        }
    
    def num_completed_sets(self, agent):
        board_prop = self.agent_board_prop if agent else self.opponent_board_prop
        count = sum(1 for a, b in zip(board_prop, self.color_to_complete_set) if a == b)
        return count
    
    def draw(self):
        return len(self.agent_hand) == 0 and len(self.opponent_hand) == 0
    
    def game_over(self, agent):
        '''
        Check if the game is over (Win or Tie)
        '''
        return self.num_completed_sets(agent) >= 2

    def calculate_rent(self, agent):
        '''
        Calculate the rent for the opponent's properties, then take the max
        '''

        rent_options = self.rent_prices[np.arange(10), self.agent_board_prop.astype(int) if not agent else self.opponent_board_prop.astype(int)]
        return np.max(rent_options)
    
    def play_rent_action(self, rent_value, agent):
        current_rent = 0
        card_values = np.array([val['value'] for _, val in self.deck.items()])
        card_values_weighted = np.array([val['value'] * 4 if card.startswith('P_') else val['value'] for card, val in self.deck.items() if val['prop_color'] != 'Any'])
        while current_rent < rent_value and len(self.agent_hand if not agent else self.opponent_hand) > 0:
            cash_and_props_on_board_weights = np.append(np.where(self.opponent_board_cash != 0,1,0) if agent else np.where(self.agent_board_cash != 0,1,0), self.opponent_board_prop if agent else self.agent_board_prop)
            cash_and_props_on_board_weights = np.multiply(cash_and_props_on_board_weights, card_values_weighted)
            cash_and_props_on_board_weights = np.where(cash_and_props_on_board_weights == 0, np.inf, cash_and_props_on_board_weights)
            min_index = np.argmin(cash_and_props_on_board_weights)
            if min_index < len(cash_and_props_on_board_weights) - len(self.color_to_index):
                card_to_give_up = min_index
                if agent:
                    self.opponent_board_cash[card_to_give_up] -= 1
                    self.agent_board_cash[card_to_give_up] += 1
                else:
                    self.agent_board_cash[card_to_give_up] -= 1
                    self.opponent_board_cash[card_to_give_up] += 1
            else:
                card_to_give_up = min_index - (len(cash_and_props_on_board_weights) - len(self.color_to_index))
                if agent:
                    self.opponent_board_prop[card_to_give_up] -= 1
                    self.agent_board_prop[card_to_give_up] += 1
                else:
                    self.agent_board_prop[card_to_give_up] -= 1
                    self.opponent_board_prop[card_to_give_up] += 1
            current_rent += card_values[min_index]
    
    def draw_card(self, agent):
        '''
        Draw a card from the deck using weighted random sampling based on quantities,
        only if there are cards remaining in the deck.
        Returns the index of the selected card and updates the deck quantities.
        '''
        # get the cards and their weights
        if np.sum(self.deck_quantities) > 0:
            cards = self.deck.keys()
            weights = self.deck_quantities / np.sum(self.deck_quantities)
            # randomly select a card
            selected_idx = random.choices(range(len(cards)), weights=weights, k=1)[0]

            # lower the quantity of the selected card by one 
            self.deck_quantities[selected_idx] -= 1
            card_drawn = list(cards)[selected_idx]
            
            self.agent_hand.append(card_drawn) if agent else self.opponent_hand.append(card_drawn)
            return card_drawn
    
    def step(self, action, agent=True):
        done = False
        card = self.deck[action]
        rent_value = 0
        multiplier = 1
        if agent:
            self.agent_hand.remove(action)
        else:
            self.opponent_hand.remove(action)
        if card['action']:
            rent_value = self.calculate_rent(agent)
            self.play_rent_action(rent_value, agent)
            multiplier = 2
        elif card['prop_color']:
            # property card
            if agent:
                self.agent_board_prop[self.color_to_index[card['prop_color']] - 1] += 1
            else:
                self.opponent_board_prop[self.color_to_index[card['prop_color']] - 1] += 1
        else:
            # cash card
            if agent:
                self.agent_board_cash[self.cash_to_index[action] - 1] += 1
            else:
                self.opponent_board_cash[self.cash_to_index[action] - 1] += 1

        # check if the game is over
        if self.draw() or self.game_over(agent):
            done = True
        reward = self.evaluate(agent)
        return self.get_observation(), reward * multiplier, done
    
    def evaluate(self, agent):
        if self.game_over(agent):
            return 10000 if agent else -10000
        if self.draw():
            return 0
        score = 0

        completed_sets = self.num_completed_sets(agent)
        score += completed_sets * 500

        for i in range(len(self.agent_board_prop if agent else self.opponent_board_prop)):
            required = self.color_to_complete_set[i]
            owned = self.agent_board_prop[i] if agent else self.opponent_board_prop[i]
            if owned > 0:
                score += (owned / required) * 100
        
        total_cash = sum(((self.agent_board_cash[i] if agent else self.opponent_board_cash[i]) * val) for i, val in enumerate([1, 2, 3, 4, 5, 10]))
        score += np.log1p(total_cash) * 20

        return score if agent else -score

    def select_random_action(self, agent):
        random_action_index = random.randint(0, len(self.agent_hand if agent else self.opponent_hand) - 1)
        random_action = self.agent_hand[random_action_index] if agent else self.opponent_hand[random_action_index]
        return random_action