# Monopoly_Deal_Agent
A set of algorithms designed to play Monopoly Deal

Algorithms
- Minimax
- Monte Carlo Tree Search
- Reinforcement Learning

# Computational Representation:

## Game States

The state of the game is defined by…
The cards in the agent’s hand (0-7)
All cards placed on the board by the agent
All cards placed on the board by the Opponent

## Inputs: 

Agent’s Hand: [card #1, card #2, … card #n]
Board Representation (One for the agent, one for the opponent) 

Example board:
[Cash: [1, 3, 4, 10],
Property Green: [0, 0, 0],
Property Dark Blue : [0, 0, 0],
Property Brown: [0, 0, 0],
Property Light Blue: [0, 0, 0],
Property Orange: [0, 0, 0],
Property Pink: [0, 0, 0],
Property Black: [0, 0, 0],
Property Red: [0, 0, 0],
Property Tan: [0, 0, 0],
Property Yellow: [0, 0, 0]]

Each property vector represents: [# of played properties for the given color (int, 0-4), Is there a house played? (bool, 0 or 1), Is there a hotel played? (bool, 0 or 1)]

## Outputs:
 
Each action returns the updated agent’s hand and board state

## Observations:
The cards in the agent’s hand
The current board representation
Actions

## During the Agent Turns:
Place a card onto the live board 
Cash 
Property
Wild Property 
House / Hotel
Play an Action card 
Dealbreaker (take a full set from opposing player)
Debt collector (take 5m from opposing)
Forced Deal (swap property with opposing player)
Birthday (collect 2m from opposing player)
Pass Go (collect 2 new cards)
Sly Deal (take one property from opposing player)
Rent (charge opposing player rent based on agent’s properties)

## During Payment to other players:
Choose cards to give to opposing player as payment
Prioritize cash over properties

## Rewards (To be used in RL): 

Win the game (complete 3 sets): +100
Place Cash: + value of Cash 
Take money from opposing Player: + value of cash x 2
Place property: value of property x (1 + % of set completed)
Steal property: value of property x (2 + % of set completed)
Complete Set: +10
