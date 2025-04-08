import gym 
from gym import spaces 
import numpy as np 
import random
import copy

'''
Limitations on this representation:
- No wild properties
- Only action card is rent

MADE SPECIFIC CHANGES TO THE GAME REPRESENTATION TO BE SUITABLE FOR RL

'''


class MonopolyDealEnv(gym.Env):
    def __init__(self):
        super(MonopolyDealEnv, self).__init__()

        cash_opt = 7
        max_prop = 5
        hand_opt = 18

        self.action_space = spaces.Discrete(10)
        self.action_mask = np.ones(10, dtype=bool)
        self.observation_space = spaces.Dict({
            "Agent hand": spaces.MultiDiscrete([hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt]),

            "Agent Board": spaces.MultiDiscrete([max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop]),
            "Opponent Board": spaces.MultiDiscrete([max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop]),

            "Agent Cash": spaces.MultiDiscrete([cash_opt, cash_opt, cash_opt, cash_opt, cash_opt, cash_opt]),
            "Opponent Cash": spaces.MultiDiscrete([cash_opt, cash_opt, cash_opt, cash_opt, cash_opt, cash_opt]),
            'Turn': spaces.Discrete(6),
            'action_mask': spaces.Box(low=0, high=1, shape=(10,), dtype=np.int8)
        })


        self.state = {
            "Agent hand": np.zeros(10, dtype=int),
            "Agent Board": np.zeros(10, dtype=int),  
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
        # self.color_to_complete_set = np.array([2, 1, 1, 2, 2, 2, 3, 2, 1, 2])
        
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
            0: {'name': 'Nothing', 'action': None, 'value': 0, 'prop_color': None},
            1: {'name': 'One_Cash', 'action': None, 'value': 1, 'prop_color': None},
            2: {'name': 'Two_Cash', 'action': None, 'value': 2, 'prop_color': None},
            3: {'name': 'Three_Cash', 'action': None, 'value': 3, 'prop_color': None},
            4: {'name': 'Four_Cash', 'action': None, 'value': 4, 'prop_color': None},
            5: {'name': 'Five_Cash', 'action': None, 'value': 5, 'prop_color': None},
            6: {'name': 'Ten_Cash', 'action': None, 'value': 10, 'prop_color': None},
            
            # Property cards (indices 1-10, matching your color_to_index)
            7: {'name': 'P_Green', 'action': None, 'value': 4, 'prop_color': 'P_Green'},
            8: {'name': 'P_DBlue', 'action': None, 'value': 4, 'prop_color': 'P_DBlue'},
            9: {'name': 'P_Brown', 'action': None, 'value': 1, 'prop_color': 'P_Brown'},
            10: {'name': 'P_LBlue', 'action': None, 'value': 1, 'prop_color': 'P_LBlue'},
            11: {'name': 'P_Orange', 'action': None, 'value': 2, 'prop_color': 'P_Orange'},
            12: {'name': 'P_Pink', 'action': None, 'value': 2, 'prop_color': 'P_Pink'},
            13: {'name': 'P_Black', 'action': None, 'value': 2, 'prop_color': 'P_Black'},
            14: {'name': 'P_Red', 'action': None, 'value': 3, 'prop_color': 'P_Red'},
            15: {'name': 'P_Tan', 'action': None, 'value': 2, 'prop_color': 'P_Tan'},
            16: {'name': 'P_Yellow', 'action': None, 'value': 3, 'prop_color': 'P_Yellow'},
            
            # Action cards
            17: {'name': 'Rent', 'action': 'rent', 'value': 3, 'prop_color': 'Any'},
        }
    

        # define the quantities of each card in the deck
        self.deck_quantities = np.array([0, 6, 5, 3, 3, 2, 1, 3, 2, 2, 3, 3, 3, 4, 3, 2, 3, 2])

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
        self.deck_quantities = np.array([0, 6, 5, 3, 3, 2, 1, 3, 2, 2, 3, 3, 3, 4, 3, 2, 3, 2])

        # draw 5 cards to start the game
        self.draw_card(True)
        self.draw_card(True)
        self.draw_card(True)
        self.draw_card(True)
        self.draw_card(True)
        # opponent draws 5 cards to start the game
        self.draw_card(False)
        self.draw_card(False)
        self.draw_card(False)
        self.draw_card(False)
        self.draw_card(False)

        # Update action mask
        self.action_mask = self.get_action_mask()
        # Add action mask to state
        self.state['action_mask'] = self.action_mask.astype(np.int8)
        
            
        return self.state


    def get_observation(self):
        '''
        Get the observation of the game
        '''
        return self.state
    
    def num_completed_sets(self, agent):

        board_prop = self.state["Agent Board"] if agent else self.state["Opponent Board"]
        count = sum(1 for a, b in zip(board_prop, self.color_to_complete_set) if a >= b)

        return count
    

    def game_over(self, agent):
        '''
        Check if the game is over
        '''
        return self.num_completed_sets(agent) >= 3
    
    
    def draw(self):
        return sum(self.deck_quantities) == 0
            

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
        
        return None

    
    
    def step(self, action, update_state=False):
        agent = self.state['Turn'] < 3
        done = False
        reward = 0
        sets = self.num_completed_sets(agent)

        if not agent:
            # For opponent, only choose from valid actions
            valid_actions = np.where(self._opponent_hand != 0)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                action = 0  # Default if no valid actions
        else:
            # Check if the action is valid for the agent
            self.action_mask = self.get_action_mask()
            if not self.action_mask[action]:
                # If invalid action, give a negative reward
                reward -= 10

        # Get the card that's being played BEFORE removing it from hand
        card_to_play = self.state["Agent hand"][action] if agent else self._opponent_hand[action]
        card = self.deck[card_to_play]
        
        # Remove card from hand by setting to 0
        if agent:
            self.state["Agent hand"][action] = 0  
        else:
            self._opponent_hand[action] = 0  

        if card['name'] == 'Nothing':
            pass

        # if card is an action card
        elif card['action']:
            rent_value = self.rent(agent)
            reward += card['value']
            # add in RENT FUNCTION HERE AHAHAHAHAHAHAHAHAHAH

        # if card is a property card
        elif card['prop_color']:
            if agent:
                self.state["Agent Board"][self.color_to_index[card['prop_color']] - 1] += 1
                reward += card['value'] * 20
            else:
                self.state["Opponent Board"][self.color_to_index[card['prop_color']] - 1] += 1

        # if card is a cash card
        else:
            if agent:
                self.state["Agent Cash"][self.cash_to_index[card['name']] - 1] += 1
                reward += card['value'] * 10
            else:
                self.state["Opponent Cash"][self.cash_to_index[card['name']] - 1] += 1

        if self.num_completed_sets(agent) > sets:
            reward += 500

        # check if the game is over
        if self.game_over(agent):
            done = True
            reward += 10000
            if agent:
                print("Agent wins")
            else:
                print("Opponent wins")
        elif self.draw():
            done = True
            print("Draw")

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


        # check the number of cards remaining and discard if over 7
        cards_left = np.count_nonzero(self.state["Agent hand"]) if agent else np.count_nonzero(self._opponent_hand)
        if (cards_left > 7 & (self.state['Turn'] == 2 or self.state['Turn'] == 5)): 

            if agent:
                self.state["Agent hand"][8] = 0 
                self.state["Agent hand"][9] = 0 
            else:
                self._opponent_hand[8] = 0 
                self._opponent_hand[9] = 0 


        #check if a player has any cards at end of turn, if not draw 5
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


        # Update action mask after step
        self.action_mask = self.get_action_mask()
        self.state['action_mask'] = self.action_mask.astype(np.int8)

        return self.state, reward if agent else 0, done, info
    
    def get_action_mask(self):
        """
        Returns a boolean mask indicating which actions are valid.
        An action is valid if the corresponding index in Agent's hand has a non-zero value.
        """
        agent_turn = self.state['Turn'] < 3
        if agent_turn:
            return self.state["Agent hand"] != 0
        else:
            return self._opponent_hand != 0
    

