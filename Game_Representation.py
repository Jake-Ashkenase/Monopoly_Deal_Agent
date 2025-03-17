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
            {'One_Cash': {'action' : None, 'value' : 1, 'prop_color': None}},
            {'Two_Cash': {'action' : None, 'value' : 2, 'prop_color': None}},
            {'Three_Cash': {'action' : None, 'value' : 3, 'prop_color': None}},
            {'Four_Cash': {'action' : None, 'value' : 4, 'prop_color': None}},
            {'Five_Cash': {'action' : None, 'value' : 5, 'prop_color': None}},
            {'Ten_Cash': {'action' : None, 'value' : 10, 'prop_color': None}},
            {'P_Green': {'action' : None, 'value' : 4, 'prop_color': 'P_Green'}},
            {'P_DBlue': {'action' : None, 'value' : 4, 'prop_color': 'P_DBlue'}},
            {'P_Brown': {'action' : None, 'value' : 1, 'prop_color': 'P_Brown'}},
            {'P_LBlue': {'action' : None, 'value' : 1, 'prop_color': 'P_LBlue'}},
            {'P_Orange': {'action' : None, 'value' : 2, 'prop_color': 'P_Orange'}},
            {'P_Pink': {'action' : None, 'value' : 2, 'prop_color': 'P_Pink'}},
            {'P_Black': {'action' : None, 'value' : 2, 'prop_color': 'P_Black'}},
            {'P_Red': {'action' : None, 'value' : 3, 'prop_color': 'P_Red'}},
            {'P_Tan': {'action' : None, 'value' : 2, 'prop_color': 'P_Tan'}},
            {'P_Yellow': {'action' : None, 'value' : 3, 'prop_color': 'P_Yellow'}},
            {'House': {'action' : None, 'value' : 3, 'prop_color': 'Set'}},
            {'Hotel': {'action' : None, 'value' : 4, 'prop_color': 'Set'}},

            # leaving out wild properties for the simplest representation of the game

            {'Rent': {'action' : Rent, 'value' : 3, 'prop_color': 'Any'}},
        }

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
        self.actions = ['CARD1', 'CARD2', 'CARD3', 'CARD4', 'CARD5', 'CARD6', 'CARD7']
        self.action_space = spaces.Discrete(4) # 4 actions: buy, sell, pass, end turn 

        def Rent(self):
            '''
            Calculate the rent for the opponent's properties, then take the max
            '''
            rent_options = self.rent_prices[np.arange(10), self.agent_board_prop[:, 0]]
            return np.max(rent_options)
        
