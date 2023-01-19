from players import learning, RLPlayer, HumanPlayer, SemiRandomPlayer
from main import RandomPlayer
from quarto.objects import Quarto

# learn G
alpha = 0.1
random_factor = 0.2
gamma = 0.7
# G = learning(alpha, random_factor, gamma)

# save G
# f = open(f'record/reward_dictionary_alpha{round(alpha*100)}_random{round(random_factor*100)}_gamma{round(gamma*100)}.txt','w')
# f.write(str(G))
# f.close()

# load G
with open(f'record/reward_dictionary_alpha{round(alpha*100)}_random{round(random_factor*100)}_gamma{round(gamma*100)}.txt', 'r') as f:
    G = eval(f.readline())


win = 0
lose = 0
for i in range(100):
    game = Quarto()
    game.set_players((RLPlayer(game, G), SemiRandomPlayer(game)))
    # game.set_players((RandomPlayer(game), RLPlayer(game, G)))

    winner = game.run()

    if winner == 0:
        win += 1
    elif winner == 1:
        lose += 1

print(f'win: {win}')
print(f'lose: {lose}')
