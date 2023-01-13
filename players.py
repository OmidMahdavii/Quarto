from quarto.objects import Player, Quarto
from main import RandomPlayer
import logging
import numpy as np
from itertools import product
from copy import deepcopy


class HumanPlayer(Player):
    """
    Human player
    """

    def __init__(self, quarto: Quarto) -> None:
        super().__init__(quarto)

    def choose_piece(self) -> int:
        selected = False
        while not selected:
            try:
                selected_piece = int(input('Choose a piece index: '))
                assert 0 <= selected_piece <= 15
                selected = True
            except:
                logging.info('Entered index is not valid')
        return selected_piece
        
    def place_piece(self) -> tuple[int, int]:
        selected = False
        while not selected:
            try:
                x, y = input('Choose a place on the board: ').split(',')
                x, y = int(x), int(y)
                assert (0, 0) <= (x, y) <= (3, 3)
                selected = True
            except:
                logging.info('Entered position is not valid')
        return x, y


class RLPlayer(Player):
    """
    Player developed by the reinforcement learning process
    """

    def __init__(self, quarto: Quarto, G: dict) -> None:
        super().__init__(quarto)
        self.quarto = quarto
        self.G = G

    def choose_piece(self) -> int:                
        # selecting the piece with the minimum average reward:
        possible_actions = cook_status(self.quarto)['possible_actions']
        reward_dict = dict()
        for a in possible_actions:
            quarto_copy = deepcopy(self.quarto)
            quarto_copy.select(a[1])
            quarto_copy.place(*a[0])
            next_state = tuple(quarto_copy.get_board_status().ravel())
            if next_state in self.G:
                reward = self.G[next_state]
            else:
                # if the state is not analyzed before, the reward corresponding to the most similar more_crowded state
                # is replaced
                placed_elements = [e for e in next_state if e != -1]
                similar_states = []
                for g in self.G:
                    if not all(item in g for item in placed_elements):
                        break
                    similarity = sum(g[i] == next_state[i] for i in range(16))
                    similar_states.append((g, similarity))
                fake_state = max(similar_states, key=lambda x: x[1])[0]
                reward = self.G[fake_state]
            if a[1] in reward_dict:
                reward_dict[a[1]].append(reward)
            else:
                reward_dict[a[1]] = [reward]
        
        return min(reward_dict.keys(), key=lambda i: np.mean(reward_dict[i]))


    def place_piece(self) -> tuple[int, int]:   
        # selecting the spot corresponding to the best state
        possible_spots = cook_status(self.quarto)['possible_spots']
        spot_reward = dict()
        for s in possible_spots:
            quarto_copy = deepcopy(self.quarto)
            quarto_copy.place(*s)
            next_state = tuple(quarto_copy.get_board_status().ravel())
            if next_state in self.G:
                reward = self.G[next_state]
            else:
                # if the state is not analyzed before, the reward corresponding to the most similar more_crowded state
                # is replaced
                placed_elements = [e for e in next_state if e != -1]
                similar_states = []
                for g in self.G:
                    if not all(item in g for item in placed_elements):
                        break
                    similarity = sum(g[i] == next_state[i] for i in range(16))
                    similar_states.append((g, similarity))
                fake_state = max(similar_states, key=lambda x: x[1])[0]
                reward = self.G[fake_state]
            spot_reward[s] = reward
        return max(spot_reward.keys(), key=lambda x: spot_reward[x])


def cook_status(quarto: Quarto) -> dict:
    cooked = dict()

    possible_pieces = [p for p in range(0, 16) if p not in quarto.get_board_status()]
    cooked['possible_pieces'] = possible_pieces

    reversed_possible_spots = [s for s in product(range(4), range(4)) if quarto.get_board_status()[s] == -1]
    possible_spots = [(s[1], s[0]) for s in reversed_possible_spots]
    cooked['possible_spots'] = possible_spots

    possible_actions = tuple(product(possible_spots, possible_pieces))
    cooked['possible_actions'] = possible_actions

    return cooked


def learning(alpha=0.15, random_factor=0.2) -> dict:
    """
    reinforcement learning process used to calculate the policy
    """
    
    ## https://stackoverflow.com/questions/6284396/permutations-with-unique-values
    # class unique_element:
    #     def __init__(self,value,occurrences):
    #         self.value = value
    #         self.occurrences = occurrences

    # def perm_unique(elements):
    #     element_set=set(elements)
    #     list_unique = [unique_element(i,elements.count(i)) for i in element_set]
    #     u=len(elements)
    #     return perm_unique_helper(list_unique,[0]*u,u-1)

    # def perm_unique_helper(list_unique,result_list,d):
    #     if d < 0:
    #         yield tuple(result_list)
    #     else:
    #         for i in list_unique:
    #             if i.occurrences > 0:
    #                 result_list[d]=i.value
    #                 i.occurrences-=1
    #                 for g in  perm_unique_helper(list_unique,result_list,d-1):
    #                     yield g
    #                 i.occurrences+=1
    
    class Agent(object):
        def __init__(self, quarto: Quarto, alpha, random_factor) -> None:
            self.state_history = [(tuple(quarto.get_board_status().ravel()), 0)]  # (initial state, reward)
            self.alpha = alpha
            self.random_factor = random_factor
            self.G = {}
            self.G[tuple(range(16))] = 0

        # def init_reward(self):
            # initialize G as a dictionary with keys=((x, y), piece) and values=reward
            # possible_states = cook_status(quarto)['possible_states']
            # for i in possible_states:
            #     self.G[i] = np.random.uniform(low=0.1, high=1.0)
            # for i in range(16):
            #     for j in range(16):
            #         temp = []
            #         for x in range(16):
            #             if x == j:
            #                 temp.append(-1)
            #             else:
            #                 temp.append(i)
            #         self.G[temp] = 0                    

        def place_piece(self, possible_spots: tuple, player: RLPlayer) -> tuple[int, int]:
            selected_spot = None
            randomN = np.random.random()
            if randomN < self.random_factor:
                # select a random spot
                selected_spot = possible_spots[np.random.choice(len(possible_spots))]
            else:
                # if exploiting, gather all possible actions and choose one with the highest G (reward)
                selected_spot = player.place_piece()
            return selected_spot

        def choose_piece(self, possible_pieces: tuple, player: RLPlayer) -> int:
            selected_piece = None
            randomN = np.random.random()
            if randomN < self.random_factor:
                # select a random piece
                selected_piece = possible_pieces[np.random.choice(len(possible_pieces))]
            else:
                # if exploiting, gather all possible pieces and choose one with the lowest average reward
                selected_piece = player.choose_piece()
            return selected_piece

        def update_state_history(self, state: tuple, reward: float) -> None:
            self.state_history.append((state, reward))

        def learn(self) -> None:
            target = 0.0

            for prev, reward in reversed(self.state_history):
                self.G[prev] = self.G[prev] + self.alpha * (target - self.G[prev]) if prev in self.G else self.alpha * target
                target += reward

            self.state_history = []

            self.random_factor -= 10e-6  # decrease random factor each episode of play

    quarto = Quarto()
    agent = Agent(quarto, alpha, random_factor)

    for i in range(5000):
            quarto_copy = deepcopy(quarto)
            RL_player = RLPlayer(quarto_copy, agent.G)
            random_player = RandomPlayer(quarto_copy)
            while not quarto_copy.check_finished():
                selected_piece = agent.choose_piece(cook_status(quarto_copy)['possible_pieces'], RL_player)
                # quarto_copy.print()
                # print(f'agent chose {selected_piece}')
                # input('')
                quarto_copy.select(selected_piece)
                previous_state = tuple(quarto_copy.get_board_status().ravel())
                
                select_ok = False
                while not select_ok:
                    select_ok = quarto_copy.place(*random_player.place_piece())
                
                new_state = tuple(quarto_copy.get_board_status().ravel())
                
                if not quarto_copy.check_winner():
                    # give a -20 reward to the state before the winning state
                    agent.update_state_history(previous_state, -20.0)
                    # give a 10 reward to the winning state
                    agent.update_state_history(new_state, 10.0)
                    break
                
                elif quarto_copy.check_finished():
                    # give a 3 reward to the state before the draw state
                    agent.update_state_history(previous_state, 3.0)
                    # give a 0 reward to the last state if it is a tie
                    agent.update_state_history(new_state, 0.0)
                    break                
                
                select_ok = False
                while not select_ok:
                    select_ok = quarto_copy.select(random_player.choose_piece())
                
                selected_spot = agent.place_piece(cook_status(quarto_copy)['possible_spots'], RL_player)
                # quarto_copy.print()
                # print(f'agent chose {selected_spot}')
                # input('')
                previous_state = new_state
                quarto_copy.place(*selected_spot)
                new_state = tuple(quarto_copy.get_board_status().ravel())
                 
                if not quarto_copy.check_winner():
                    # give a -20 reward to the state before the winning state
                    agent.update_state_history(previous_state, -20.0)
                    # give a 10 reward to the winning state
                    agent.update_state_history(new_state, 10.0)
                    break

                elif quarto_copy.check_finished():
                    # give a 3 reward to the state before the draw state
                    agent.update_state_history(previous_state, 3.0)
                    # give a 0 reward to the last state if it is a tie
                    agent.update_state_history(new_state, 0.0)
                    break 
                
                else:
                    # give a 3 reward to the previous if the game is not finished
                    agent.update_state_history(previous_state, 3.0)

            agent.learn()
            if i % 100 == 0:
                print(f'{i}: {len(agent.G.keys())}')
            # print(agent.G)
            # exit()
        
    return agent.G