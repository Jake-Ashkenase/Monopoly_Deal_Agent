import gym 
from gym import spaces 
import numpy as np 
import random
import copy

'''
Limitations on this representation:
- No wild properties
- Action cards: rent and sly_deal

MADE SPECIFIC CHANGES TO THE GAME REPRESENTATION TO BE SUITABLE FOR RL

'''


class MonopolyDealEnv(gym.Env):
    def __init__(self):
        super(MonopolyDealEnv, self).__init__()

        cash_opt = 7
        max_prop = 5
        hand_opt = 19

        # Change to MultiDiscrete - first dimension for card selection, second for target selection
        self.action_space = spaces.MultiDiscrete([10, 10])  # [card index, property to steal]
        self.card_action_mask = np.ones(10, dtype=bool)
        self.property_action_mask = np.ones(10, dtype=bool)
        
        self.observation_space = spaces.Dict({
            "Agent hand": spaces.MultiDiscrete([hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt, hand_opt]),

            "Agent Board": spaces.MultiDiscrete([max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop]),
            "Opponent Board": spaces.MultiDiscrete([max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop, max_prop]),

            "Agent Cash": spaces.MultiDiscrete([cash_opt, cash_opt, cash_opt, cash_opt, cash_opt, cash_opt]),
            "Opponent Cash": spaces.MultiDiscrete([cash_opt, cash_opt, cash_opt, cash_opt, cash_opt, cash_opt]),
            'Turn': spaces.Discrete(6),
            'card_mask': spaces.Box(low=0, high=1, shape=(10,), dtype=np.int8),
            'property_mask': spaces.Box(low=0, high=1, shape=(10,), dtype=np.int8)
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
            18: {'name': 'Sly_Deal', 'action': 'sly_deal', 'value': 3, 'prop_color': None},
        }
    

        # define the quantities of each card in the deck
        self.deck_quantities = np.array([0, 6, 5, 3, 3, 2, 1, 3, 2, 2, 3, 3, 3, 4, 3, 2, 3, 2, 2])

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
        self.deck_quantities = np.array([0, 6, 5, 3, 3, 2, 1, 3, 2, 2, 3, 3, 3, 4, 3, 2, 3, 2, 2])

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

        # Update action masks
        self.card_action_mask, self.property_action_mask = self.get_action_masks()
        # Add action masks to state
        self.state['card_mask'] = self.card_action_mask.astype(np.int8)
        self.state['property_mask'] = self.property_action_mask.astype(np.int8)
        
            
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
        bill_values = [1, 2, 3, 4, 5, 10]

        prop_values = [4, 4, 1, 1, 2, 2, 2, 3, 2, 3]

        # find the needed rent to collect
        rent_options = self.rent_prices[np.arange(10), self.state["Agent Board"]] if agent else self.rent_prices[np.arange(10), self.state["Opponent Board"]]
        rent_to_collect = np.max(rent_options)

        # find which board to take from
        board_to_take_from = self.state["Opponent Board"] if agent else self.state["Agent Board"]
        cash_to_take_from = self.state["Opponent Cash"] if agent else self.state["Agent Cash"]

        # collect cash/properties until the rent is paid
        money_owed = rent_to_collect
        while money_owed > 0:
            cash_values = [bill_values[i] if count > 0 else 0 for i, count in enumerate(cash_to_take_from)] 
            board_values = [prop_values[i] if count > 0 else 0 for i, count in enumerate(board_to_take_from)]

            options = cash_values + board_values

            # if there is nothing left to pay with, break
            if max(options) == 0:
                break

            # Filter out 0 values and find the minimum non-zero option
            non_zero_options = [x for x in options if x > 0]
            if not non_zero_options:  # if all options are 0
                break
            min_value = min(non_zero_options)
            min_index = options.index(min_value)

            if min_index < len(cash_values):
                # Remove from one player's cash and add to other player's cash
                cash_to_take_from[min_index] -= 1
                if agent:
                    self.state["Agent Cash"][min_index] += 1
                else:
                    self.state["Opponent Cash"][min_index] += 1
            else:
                # Remove from one player's board and add to other player's board
                board_index = min_index - len(cash_values)
                board_to_take_from[board_index] -= 1
                if agent:
                    self.state["Agent Board"][board_index] += 1
                else:
                    self.state["Opponent Board"][board_index] += 1

            money_owed -= min_value

        # update the boards based on the changes 
        if agent:
            self.state["Opponent Cash"] = cash_to_take_from
            self.state["Opponent Board"] = board_to_take_from
        else:
            self.state["Agent Cash"] = cash_to_take_from
            self.state["Agent Board"] = board_to_take_from

        return np.max(rent_options)
    
    def sly_deal(self, agent, property_idx=None):
        '''
        Steal a property from the opponent's board
        If agent is True, steal from opponent
        If agent is False, steal from agent
        property_idx is the index of the property to steal (0-9 corresponding to property colors)
        '''
        source_board = "Opponent Board" if agent else "Agent Board"
        target_board = "Agent Board" if agent else "Opponent Board"
        
        # If no property is selected or if there's nothing to steal
        if property_idx is None:
            # If no property is specified, handle differently based on who's playing
            if not agent:  # Opponent's turn
                # Randomly select a property from agent's board
                available_props = np.where(self.state["Agent Board"] > 0)[0]
                if len(available_props) > 0:
                    property_idx = np.random.choice(available_props)
                else:
                    return 0  # No properties to steal
            else:
                # Agent didn't specify a property (shouldn't happen due to policy)
                return 0
                
        # Check if there's a property at the selected index
        if self.state[source_board][property_idx] > 0:
            # Transfer the property
            self.state[source_board][property_idx] -= 1
            self.state[target_board][property_idx] += 1
            # Return the value of the stolen property
            return self.deck[7 + property_idx]['value']
        
        return 0  # No property was stolen
    
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
        # Unpack the action into card_idx and property_idx
        if isinstance(action, (list, tuple, np.ndarray)):
            card_idx, property_idx = action
        else:
            card_idx = action
            property_idx = None
            
        agent = self.state['Turn'] < 3
        done = False
        reward = 0
        sets = self.num_completed_sets(agent)
        game_status = 'Playing'

        if not agent:
            # For opponent, only choose from valid actions
            valid_actions = np.where(self._opponent_hand != 0)[0]
            if len(valid_actions) > 0:
                card_idx = np.random.choice(valid_actions)
            else:
                card_idx = 0  # Default if no valid actions
                
            # For sly_deal, choose randomly from agent's properties
            if self._opponent_hand[card_idx] == 18:  # If sly_deal card
                valid_properties = np.where(self.state["Agent Board"] > 0)[0]
                if len(valid_properties) > 0:
                    property_idx = np.random.choice(valid_properties)
                else:
                    property_idx = None  # No properties to steal
        else:
            # Check if the card action is valid for the agent
            card_mask, property_mask = self.get_action_masks()
            if not card_mask[card_idx]:
                raise ValueError(f"Invalid card action {card_idx} selected. Card mask: {card_mask}")
            

            # Just check if we're playing sly_deal with no available properties
            card_to_play = self.state["Agent hand"][card_idx]
            if card_to_play == 18 and not np.any(property_mask):
                # No properties to steal, so property_idx doesn't matter
                property_idx = None

        # Get the card that's being played BEFORE removing it from hand
        card_to_play = self.state["Agent hand"][card_idx] if agent else self._opponent_hand[card_idx]
        card = self.deck[card_to_play]
        
        # Remove card from hand by setting to 0
        if agent:
            self.state["Agent hand"][card_idx] = 0  
        else:
            self._opponent_hand[card_idx] = 0  

        if card['name'] == 'Nothing':
            pass

        # if card is an action card
        elif card['action']:
            if card['action'] == 'rent':
                rent_value = self.rent(agent)
                reward += rent_value 
            elif card['action'] == 'sly_deal':
                stolen_value = self.sly_deal(agent, property_idx)
                if agent:
                    reward += stolen_value 
                else:
                    reward -= stolen_value   

        # if card is a property card
        elif card['prop_color']:
            if agent:
                self.state["Agent Board"][self.color_to_index[card['prop_color']] - 1] += 1
                reward += card['value'] * 1.5
            else:
                self.state["Opponent Board"][self.color_to_index[card['prop_color']] - 1] += 1

        # if card is a cash card
        else:
            if agent:
                self.state["Agent Cash"][self.cash_to_index[card['name']] - 1] += 1
                reward += card['value'] 
            else:
                self.state["Opponent Cash"][self.cash_to_index[card['name']] - 1] += 1

        if self.num_completed_sets(agent) > sets:
            reward += 50

        # check if the game is over
        if self.game_over(agent):
            done = True
            if agent:
                game_status = 'Agent wins'
                reward += 100
            else:
                game_status = 'Opponent wins'
                reward -= 100
        elif self.draw():
            done = True
            game_status = 'Draw'

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
        'Game_status': game_status
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


        # Update action masks after step
        self.card_action_mask, self.property_action_mask = self.get_action_masks()
        self.state['card_mask'] = self.card_action_mask.astype(np.int8)
        self.state['property_mask'] = self.property_action_mask.astype(np.int8)

        return self.state, reward if agent else 0, done, info
    
    def get_action_masks(self):
        """
        Returns two boolean masks:
        1. card_mask: indicating which cards in hand are valid to play
        2. property_mask: indicating which properties are valid to steal (only relevant for sly_deal)
        """
        agent_turn = self.state['Turn'] < 3
        
        # Card mask - can play any card in hand
        if agent_turn:
            card_mask = self.state["Agent hand"] != 0
            
            # Check if any sly_deal cards in hand
            sly_deal_indices = np.where(self.state["Agent hand"] == 18)[0]
            
            # Property mask - can only steal properties that opponent has
            # By default, no properties can be stolen
            property_mask = np.zeros(10, dtype=bool)
            
            if len(sly_deal_indices) > 0:
                # If we have sly_deal, mark properties that can be stolen
                property_mask = self.state["Opponent Board"] > 0
        else:
            card_mask = self._opponent_hand != 0
            
            # Similar logic for opponent
            sly_deal_indices = np.where(self._opponent_hand == 18)[0]
            property_mask = np.zeros(10, dtype=bool)
            
            if len(sly_deal_indices) > 0:
                property_mask = self.state["Agent Board"] > 0
                
        return card_mask, property_mask
    

