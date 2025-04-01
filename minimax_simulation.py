from Game_Representation_Minimax import *
from Minimax import *

# simulate 100 games, one turn at a time
game = MonopolyDealMinimaxEnv()
agent_wins = 0
opp_wins = 0
ties = 0
for i in range(100):
    if i % 20 == 0:
        print(i)
    obs = game.reset()
    done = False
    while not done:
        minimax = MonopolyDealMinimax(obs['agent_hand'], obs['agent_board_prop'], obs['agent_board_cash'], obs['opponent_hand'], obs['opponent_board_prop'], obs['opponent_board_cash'], obs['deck_quantities'])
        action = minimax.choose_best_action(agent=True)
        # print(f'Agent Hand: {obs["agent_hand"]}, Agent Board: {obs["agent_board_prop"]}, move: {action}')
        obs, reward, done = game.step(action, True)
        if done:
            if reward == 0:
                ties += 1
            else:
                agent_wins +=1
            continue

        game.draw_card(True)
        action = game.select_random_action(agent=False)
        # print(f'Opp Hand: {obs["opponent_hand"]}, Opp Board: {obs["opponent_board_prop"]}, move: {action}')
        obs, reward, done = game.step(action, False)
        if done:
            if reward == 0:
                ties += 1
            else:
                opp_wins +=1
        else:
            game.draw_card(False)
    # print(obs['agent_board_prop'])
    # print(obs['opponent_board_prop'])
print(f'Agent Wins: {agent_wins}')
print(f'Opponent Wins: {opp_wins}')
print(f'Draws: {ties}')