from players import learning, RLPlayer
from main import RandomPlayer
from quarto.objects import Quarto

# learn G
alpha = 0.15
random_factor = 0.3
# G = learning(alpha, random_factor)

# load G
with open(f'record/alpha{round(alpha*100)}_random{round(random_factor*100)}_result{100}.txt', 'r') as f:
    G = eval(f.readline())

# print(G.values())
# exit()

win = 0
lose = 0
for i in range(30):
    game = Quarto()
    # game.set_players((RLPlayer(game, G), RandomPlayer(game)))
    game.set_players((RandomPlayer(game), RLPlayer(game, G)))

    print(i)
    winner = game.run()

    if winner == 0:
        win += 1
    elif winner == 1:
        lose += 1

print(f'win: {win}')
print(f'lose: {lose}')
result = win / 30

# save G
# f = open(f'record/alpha{round(alpha*100)}_random{round(random_factor*100)}_result{round(result*100)}.txt','w')
# f.write(str(G))
# f.close()
