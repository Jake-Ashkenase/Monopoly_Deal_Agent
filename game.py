from Game_Representation_RL import *

game = MonopolyDealEnv()
for i in range(1000):
    print(i)
    obs = game.reset()
    done = False
    while not done:
        obs, reward, done, info = game.step(np.random.randint(0, 10))
        # print(obs)
        # print(reward)
        # print(done)

    print(game.get_observation())

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