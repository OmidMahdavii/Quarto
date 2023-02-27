# Quarto

## Task

As the final project for the Computational Intelligence course, I had to program an agent to be able to play Quarto efficiently. The basic board and the match were already implemented by the professor.

## Algorithm Description

I have developed a player via a reinforcement learning process. In the following, there are some brief descriptions of the algorithm. The project contains several modules. While the *\__init__.py*, *objects.py*, and *main.py* are provided by the professor, *players.py,* and *tournament.py* are implemented by myself.

Saving the state as a list of pieces is not feasible due to a large number of possibilities. Therefore, I implemented some kind of mapping by the `state_encoder` function to limit the number of possible states. There are overall 8 characteristics for the pieces eg. tall, not tall, etc. Furthermore, on the board, there are 10 sets of combinations that could involve a match and cause a player to win (4 rows, 4 columns, and 2 diameters). The state is mapped to a tuple containing 8 inner tuples. Each inner tuple corresponds to a characteristic and has two elements. The first element is the index of the characteristic and the second one is the number of pieces sharing the characteristic in the set that has the largest number of pieces with that characteristic among the above-mentioned sets. This method surely ignores some differences between the states but it cannot miss a winning state. In this case, if the second element of any inner tuple is equal to *4*, the board is in a winning situation.

The reinforcement learning procedure requires an opponent for the agent during the learning phase. In my opinion, the `RandomPlayer` is not good enough to train the agent against it. Hence, I wrote the `SemiRandomPlayer` class. In the `place_piece` method, all the possible positions available for the selected piece are analyzed, and if there is a position that causes the player to win, it is chosen. Otherwise, a random spot is picked up. However, in the `choose_piece` method, all combinations of `(possible_spots, possible_pieces)` for the current board state are evaluated and a random piece which does not correspond to a winning state is returned. If there is a winning move for every piece, just a random piece will be selected.

The `RLPlayer` class is an upgraded version of the `SemiRandomPlayer`. In the `place_piece` method, when there is not any winning move, the position that has the highest amount in the already-saved reward dictionary ***G*** is returned. On the other hand, in the `choose_piece` method, among all the pieces that are not *forbidden*, the average of their corresponding future state reward is calculated, and the piece with the minimum average is chosen. Picking the piece corresponding to the minimum state reward is not useful since the opponent would place it at the best spot. In addition, if I was sure that the opponent is intelligent enough, it would be a better idea to select the piece which has the lowest maximum reward saved in ***G***.

With regard to the learning process, several considerations have to be mentioned:

* the `learning` function takes 3 arguments as input. `alpha` is the learning rate and determines the amount of change for the rewards after each update. `random_factor` is used to keep the balance between *exploration* and *exploitation* and defines the probability to make a random move instead of the one based on the current reward amounts. Finally, `gamma` is the *discount rate* and affects on how much the agent prefers an immediate reward. These three parameters are my model's hyper-parameters. After several evaluations, the chosen combination is the following:

  `(alpha=0.1, random_factor=0.2, gamma=0.7)`

* Initializing the state rewards are done in the `init_reward` method. According to the implemented state mapping, a random number between *0.1* and *1.0* is assigned to each possible combination of the inner tuples. Some states saved in the dictionary are not feasible on the board due to the game characteristics. Nevertheless, in order to avoid complexity, I ignored discarding them. This decision causes the reward dictionary to be a little larger and consequently initialize slower but does not affect the game's performance since the agent can make a move in a few seconds. 
* During the training, the agent plays against the `SemiRandomPlayer` and the starting player changes after each match. After each move that the agent makes, a reward is assigned to the resulting state. If the player wins, the reward is equal to *5*. If the game is finished as a tie, a *-2* reward is assigned to the final state. However, if the game is not finished yet, the algorithm waits for the opponent to make a move and then assigns the reward. In case the opponent wins the game by its move, the agent's action will get *-5* as the reward. A draw leads to *-2*, and if the game still is not ended, *0* is saved for the state.

## Initialization

```python
# import the player
from players import RLPlayer

# import the game
from quarto.objects import Quarto

# load G
with open(f'record/reward_dictionary.txt', 'r') as f:
    G = eval(f.readline())

# init the game
game = Quarto()

# init the player
my_player = RLPlayer(game, G)
```

