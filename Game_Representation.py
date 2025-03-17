import gym 
from gym import spaces 
import numpy as np 
import random 

'''
Choosing to use gym for the environment as it is made by OpenAI and is a standard way to represent 
environments in reinforcement learning.
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

        # define the different actions
        def Rent(self):
            rent_price = max(self.opponent_board_prop[0]) # need to multiple by the rent prices
            return rent_price 

        self.cards= {
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


        self.rent_prices = {
            'P_Green' : [2, 4, 7],
            'P_DBlue': [3, 8],
            'P_Brown': [1, 2],
            'P_LBlue': [1, 2, 3],
            'P_Orange': [1, 3, 5],
            'P_Pink': [1, 2, 4],
            'P_Black': [1, 2, 3, 4],
            'P_Red': [2, 3, 6],
            'P_Tan': [1, 2],
            'P_Yellow': [2, 4, 6],
        }

        # Define action space s
        self.actions = ['CARD1', 'CARD2', 'CARD3', 'CARD4', 'CARD5', 'CARD6', 'CARD7']
        self.action_space = spaces.Discrete(4) # 4 actions: buy, sell, pass, end turn 
        
