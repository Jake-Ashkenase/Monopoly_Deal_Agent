import gym 
from gym import spaces 
import numpy as np 
import random 

'''
Limitations on this representation:
- No wild properties
- Only action card is rent
'''


class MonopolyDealEnv(gym.Env):
    def __init__(self):
        super(MonopolyDealEnv, self).__init__()

        # define the hand of the agent, a max of 7 cards 
        self.agent_hand = []

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

        self.color_to_complete_set = np.array([3, 2, 2, 3, 3, 3, 4, 3, 2, 3])
        
        # define the board of the agent, with a vector for each color representing...
        #  [number of properties played, number of houses, number of hotels]
        self.agent_board_prop = np.zeros((10, 3))  # 11 colors, 3 values each
        self.agent_board_cash = []



        # define the board of the opponent, with a vector for each color representing...
        #  [number of properties played, number of houses, number of hotels]
        self.opponent_board_prop = np.zeros((10, 3))  # 11 colors, 3 values each
        self.opponent_board_cash = []

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

            'Rent': {'action' : rent, 'value' : 3, 'prop_color': 'Any'},
        }
    

        # define the quantities of each card in the deck

        self.deck_quantities = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

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
        self.actions = ['CARD1', 'CARD2', 'CARD3', 'CARD4', 'CARD5', 'CARD6', 'CARD7', 'PASS']


        def reset(self):
            '''
            Reset the game
            '''
            self.agent_hand = []
            self.agent_board_prop = np.zeros((10, 3))
            self.opponent_board_prop = np.zeros((10, 3))
            self.agent_board_cash = []
            self.opponent_board_cash = []
            self.deck_quantities = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0])

            return self.get_observation()


        def get_observation(self):
            '''
            Get the observation of the game
            '''
            return {
                'agent_hand': self.agent_hand,
                'agent_board_prop': self.agent_board_prop,
                'opponent_board_prop': self.opponent_board_prop,
                'agent_board_cash': self.agent_board_cash,
                'opponent_board_cash': self.opponent_board_cash,
            }
        
        def num_completed_sets(self):
            count = sum(1 for a, b in zip(self.agent_board_prop[:, 0], self.color_to_complete_set) if a == b)
            return count
        

        def game_over(self):
            '''
            Check if the game is over
            '''
            return self.num_completed_sets() >= 3
                

        def rent(self):
            '''
            Calculate the rent for the opponent's properties, then take the max
            '''

            rent_options = self.rent_prices[np.arange(10), self.agent_board_prop[:, 0]]
            return np.max(rent_options)
        
        def draw_card(self):
            '''
            Draw a card from the deck using weighted random sampling based on quantities.
            Returns the index of the selected card and updates the deck quantities.
            '''
            # get the cards and their weights
            cards = self.deck.keys()
            weights = self.deck_quantities / np.sum(self.deck_quantities)

            # randomly select a card
            selected_idx = random.choices(range(len(cards)), weights=weights, k=1)[0]

            # lower the quantity of the selected card by one 
            self.deck_quantities[selected_idx] -= 1
            card_drawn = cards[selected_idx]
            
            self.agent_hand.append(card_drawn)
            return card_drawn
        
        def step(self, action):
            done = False
            card = self.deck[action]
            reward = 0
            sets = self.num_completed_sets()
            
            self.agent_hand.remove(card)
            if card['action']:
                rent_value = card['action'](self)
                reward += rent_value * 2
            elif card['prop_color']:
                #property card
                self.agent_board_prop[self.color_to_index[card['prop_color']], 0] += 1
                reward += card['value']
            else:
                #cash card
                self.agent_board_cash += card['value']
                reward += card['value']

            if self.num_completed_sets() > sets:
                reward += 10

            #check if the game is over
            if self.game_over():
                done = True
                reward += 100
                return self.get_observation(), reward, done










                
        


        

        
        
        
