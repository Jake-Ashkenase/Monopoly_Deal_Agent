from Game_Representation import *

game = MonopolyDealEnv()
# obs = game.reset()
# done = False
# while not done:
#     for i in range(3):
#         try:
#             action = game.select_random_action(agent=True)
#         except:
#             continue
#         obs, reward, done = game.step(action, True)
#         print(f'Agent: {action}, {reward}')
#         game.draw_card(True)
#     for i in range(3):
#         try:
#             action = game.select_random_action(agent=False)
#         except ValueError:
#             continue
#         obs, reward, done = game.step(action, False)
#         print(f'Opponent: {action}, {reward}')
#         game.draw_card(False)
# print(game.get_observation())

def select_best_action_via_minimax(self, agent):
    bestScore = -10000
    bestAction = None
    for card in (self.agent_hand if agent else self.opponent_hand):
        self.step(card, agent) # This is changing the global variables. Make copy of board / undo move / idk how to resovle.
        # actually i think make a constructor that takes in all the current state variables and move this function into game.py,
        # instantiating a new MonopolyDealEnv() for each minimax play
        score = self.minimax(0, not agent)
        if (score > bestScore):
            bestScore = score
            bestAction = card
    return bestAction

def minimax(self, depth, agent):
    if self.game_over(agent):
        return self.rewards['Goal']
    if agent:
        bestScore = -10000
        for card in (self.agent_hand):
            self.step(card, agent)
            score = self.minimax(depth+1, not agent)
            if score > bestScore:
                bestScore = score
    else:
        bestScore = 10000
        for card in self.opponent_hand:
            self.step(card, not agent)
            score = self.minimax(depth+1, agent)
            if score < bestScore:
                bestScore = score
    return bestScore