from players import learning, RLPlayer, HumanPlayer, SemiRandomPlayer
from main import RandomPlayer
from quarto.objects import Quarto

# learn G
alpha = 0.15
random_factor = 0.2
# G = learning(alpha, random_factor)

# save G
# f = open(f'record/reward_dictionary_alpha{round(alpha*100)}_random{round(random_factor*100)}.txt','w')
# f.write(str(G))
# f.close()

# load G
with open(f'record/reward_dictionary_alpha{round(alpha*100)}_random{round(random_factor*100)}.txt', 'r') as f:
    G = eval(f.readline())

# print(G.values())
print(len(G.values()))
exit()

win = 0
lose = 0
for i in range(30):
    game = Quarto()
    game.set_players((RLPlayer(game, G), SemiRandomPlayer(game)))
    # game.set_players((RandomPlayer(game), RLPlayer(game, G)))

    print(i)
    winner = game.run()

    if winner == 0:
        win += 1
    elif winner == 1:
        lose += 1

print(f'win: {win}')
print(f'lose: {lose}')
result = win / 30

