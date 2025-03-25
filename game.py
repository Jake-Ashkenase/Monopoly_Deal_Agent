from Game_Representation import *

game = MonopolyDealEnv()

def play_game():
    done = False
    obs = game.reset()
    counter = 0
    while not done:
        agent_total_reward = 0
        opponent_total_reward = 0
        for i in range(1):
            action = game.select_random_action(agent=True)
            obs, reward, done = game.step(action, True)
            agent_total_reward += reward
            try:
                game.draw_card(True)
            except ValueError:
                print("No more cards in deck")
                # print(obs['agent_board_prop'])
                # print(f'Agent Completed Sets: {game.num_completed_sets(True)}')
        for i in range(1):
            action = game.select_random_action(agent=False)
            obs, reward, done = game.step(action, False)
            opponent_total_reward += reward
            try:
                game.draw_card(False)
            except ValueError:
                print("No more cards in deck")
        #         print(obs['opponent_board_prop'])
        #         print(f'Opponent Completed Sets: {game.num_completed_sets(False)}')
        # counter += 1
        # print(f'{counter}: {game.deck_quantities}')
    print(obs)

def select_best_action_via_minimax(agent):
        bestScore = -10000
        bestMove = None
        for card in (game.agent_hand if agent else self.opponent_hand):
            minimax()

play_game()