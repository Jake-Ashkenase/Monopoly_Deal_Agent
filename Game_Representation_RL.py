import gym 
from gym import spaces 
import numpy as np 
import random 

'''
Limitations on this representation:
- No wild properties
- Only action card is rent

MADE SPECIFIC CHANGES TO THE GAME REPRESENTATION TO BE SUITABLE FOR RL

'''


class MonopolyDealEnv(gym.Env):
    def __init__(self):
        super(MonopolyDealEnv, self).__init__()

        prop_opt = 10
        cash_opt = 6
        max_prop = 4
        action_card_opt = 1
        hand_opt = 17

        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Dict({
            "Agent hand": spaces.MultiDiscrete([hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt]),

            "Agent Board": spaces.MultiDiscrete([max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop]),
            "Opponent Board": spaces.MultiDiscrete([max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop]),

            "Agent Cash": spaces.MultiDiscrete([cash_opt, cash_opt, cash_opt, cash_opt, cash_opt, cash_opt]),
            "Opponent Cash": spaces.MultiDiscrete([cash_opt, cash_opt, cash_opt, cash_opt, cash_opt, cash_opt]),
            'Turn': spaces.Discrete(6)
        })


        self.state = {
            "Agent hand": np.zeros(10, dtype=int),  # matches MultiDiscrete([hand_opt] * 10)
            "Agent Board": np.zeros(10, dtype=int),  # matches MultiDiscrete([max_prop] * 10)
            "Opponent Board": np.zeros(10, dtype=int),
            "Agent Cash": np.zeros(6, dtype=int),
            "Opponent Cash": np.zeros(6, dtype=int),
            'Turn': 0
        }

        # Maintain the opponent's hand separately since it's not part of observation
        self._opponent_hand = np.zeros(10, dtype=int)  


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
        
        self.rewards = {
            'Goal': 100,  # complete 3 sets 
            'Cash': 0,  # needs to be the value of cash placed 
            'Steal Cash': 0*2,  # needs to be the value of cash stolen, times 2
            'Place Property': 0,  # needs to be the value of the property placed
            'Steal Property': 0*2,  # needs to be the value of property stolen, times 2
            'Set': 10
        }

        self.deck = {
            # Cash cards (indices 1-6)
            0: {'name': 'One_Cash', 'action': None, 'value': 1, 'prop_color': None},
            1: {'name': 'Two_Cash', 'action': None, 'value': 2, 'prop_color': None},
            2: {'name': 'Three_Cash', 'action': None, 'value': 3, 'prop_color': None},
            3: {'name': 'Four_Cash', 'action': None, 'value': 4, 'prop_color': None},
            4: {'name': 'Five_Cash', 'action': None, 'value': 5, 'prop_color': None},
            5: {'name': 'Ten_Cash', 'action': None, 'value': 10, 'prop_color': None},
            
            # Property cards (indices 1-10, matching your color_to_index)
            6: {'name': 'P_Green', 'action': None, 'value': 4, 'prop_color': 'P_Green'},
            7: {'name': 'P_DBlue', 'action': None, 'value': 4, 'prop_color': 'P_DBlue'},
            8: {'name': 'P_Brown', 'action': None, 'value': 1, 'prop_color': 'P_Brown'},
            9: {'name': 'P_LBlue', 'action': None, 'value': 1, 'prop_color': 'P_LBlue'},
            10: {'name': 'P_Orange', 'action': None, 'value': 2, 'prop_color': 'P_Orange'},
            11: {'name': 'P_Pink', 'action': None, 'value': 2, 'prop_color': 'P_Pink'},
            12: {'name': 'P_Black', 'action': None, 'value': 2, 'prop_color': 'P_Black'},
            13: {'name': 'P_Red', 'action': None, 'value': 3, 'prop_color': 'P_Red'},
            14: {'name': 'P_Tan', 'action': None, 'value': 2, 'prop_color': 'P_Tan'},
            15: {'name': 'P_Yellow', 'action': None, 'value': 3, 'prop_color': 'P_Yellow'},
            
            # Action cards
            16: {'name': 'Rent', 'action': 'rent', 'value': 3, 'prop_color': 'Any'},
        }
    

        # define the quantities of each card in the deck
        self.deck_quantities = np.array([6, 5, 3, 3, 2, 1, 3, 2, 2, 3, 3, 3, 4, 3, 2, 3, 2])

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


    def reset(self):
        '''
        Reset the game
        '''
        # Reset the main state dictionary
        self.state = {
            "Agent hand": np.zeros(10, dtype=int), 
            "Agent Board": np.zeros(10, dtype=int),  
            "Opponent Board": np.zeros(10, dtype=int),
            "Agent Cash": np.zeros(6, dtype=int),    
            "Opponent Cash": np.zeros(6, dtype=int),
            'Turn': 0
        }
        
        # Reset the hidden opponent hand
        self._opponent_hand = np.zeros(10, dtype=int)
        
        # Reset deck quantities
        self.deck_quantities = np.array([6, 5, 3, 3, 2, 1, 3, 2, 2, 3, 3, 3, 4, 3, 2, 3, 2])
        
        # Draw initial hands
        for _ in range(5):
            self.draw_card(True)
            self.draw_card(False)
            
        return self.state


    def get_observation(self):
        '''
        Get the observation of the game
        '''
        return self.state
    
    def num_completed_sets(self, agent):
        board_prop = self.state["Agent Board"] if agent else self.state["Opponent Board"]
        count = sum(1 for a, b in zip(board_prop, self.color_to_complete_set) if a == b)
        return count
    

    def game_over(self, agent):
        '''
        Check if the game is over
        '''
        return self.num_completed_sets(agent) >= 3
            

    def rent(self, agent):
        '''
        Calculate the rent for the opponent's properties, then take the max
        '''

        rent_options = self.rent_prices[np.arange(10), self.state["Agent Board"]] if agent else self.rent_prices[np.arange(10), self.state["Opponent Board"]]
        return np.max(rent_options)
    
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
            
            if agent:
                zero_index = np.where(self.state["Agent hand"] == 0)[0][0]  # find first zero
                self.state["Agent hand"][zero_index] = card_drawn
            else:
                zero_index = np.where(self._opponent_hand == 0)[0][0]  # find first zero
                self._opponent_hand[zero_index] = card_drawn

            return card_drawn

    
    
    def step(self, action, agent=True, update_state=False):
        print(f"stepping with action: {action}")
        print(self.state["Agent hand"])
        done = False
        card = self.deck[action]
        reward = 0
        sets = self.num_completed_sets(agent)

        #check if a player has any cards to start a turn, if not draw 5
        hand_to_check = self.state["Agent hand"] if agent else self._opponent_hand
        if np.all(hand_to_check == 0):
            self.draw_card(agent)
            self.draw_card(agent)
            self.draw_card(agent)
            self.draw_card(agent)
            self.draw_card(agent)

        # draw 2 cards to start a turn (every 3 moves)
        if self.state['Turn'] == 0:
            self.draw_card(True)
            self.draw_card(True)
        elif self.state['Turn'] == 3:
            self.draw_card(False)
            self.draw_card(False)
            
        # Remove card from hand. remove by setting to 0, 
        # drawing a card will fill the slot first (left to right)
        # the action index is the same as the index of the card in a players hand
        if agent:
            self.state["Agent hand"][action] = 0  
        else:
            self._opponent_hand[action] = 0  

        # if card is an action card
        if card['action']:
            rent_value = self.rent(agent)
            reward += card['value']
            # add in RENT FUNCTION HERE AHAHAHAHAHAHAHAHAHAH

        # if card is a property card
        elif card['prop_color']:
            if agent:
                self.state["Agent Board"][self.color_to_index[card['prop_color']] - 1] += 1
            else:
                self.state["Opponent Board"][self.color_to_index[card['prop_color']] - 1] += 1
            reward += card['value']

        # if card is a cash card
        else:
            if agent:
                self.state["Agent Cash"][self.cash_to_index[card['name']] - 1] += 1
            else:
                self.state["Opponent Cash"][self.cash_to_index[card['name']] - 1] += 1
            reward += card['value']

        if self.num_completed_sets(agent) > sets:
            reward += 10

        # check if the game is over
        if self.game_over(agent):
            done = True
            reward += 100

        # change the turn
        if self.state['Turn'] == 5:
            self.state['Turn'] = 0
        else: 
            self.state['Turn'] += 1

        info = {
        'completed_sets': self.num_completed_sets(agent),
        'opponent_sets': self.num_completed_sets(not agent),
        'agent_cash_total': sum(i * count for i, count in enumerate([1,2,3,4,5,10], 1) 
                              for j in range(self.state["Agent Cash"][i-1])),
        'opponent_cash_total': sum(i * count for i, count in enumerate([1,2,3,4,5,10], 1) 
                                 for j in range(self.state["Opponent Cash"][i-1])),
        'episode_step': getattr(self, '_steps', 0),  # You might want to add a step counter
        'is_success': done and self.num_completed_sets(agent) >= 3  # True if won by completing sets
        }

        return self.state, reward, done, info
    

