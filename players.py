from quarto.objects import Player, Quarto
from main import RandomPlayer
import logging
import numpy as np
from itertools import product, combinations
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
        self.winning_states = self.init_winning_states()

    def init_winning_states(self) -> tuple:
        not_high_pieces = list(range(8))
        high_pieces = list(range(8, 16))
        not_colored_pieces = list(range(4)) + list(range(8, 12))
        colored_pieces = list(range(4, 8)) + list(range(12, 16))
        not_solid_pieces = [0, 1, 4, 5, 8, 9, 12, 13]
        solid_pieces = [2, 3, 6, 7, 10, 11, 14, 15]
        not_square_pieces = list(range(0, 16, 2))
        square_pieces = list(range(1, 16, 2))

        all_lists = []
        all_lists.append(not_high_pieces)
        all_lists.append(high_pieces)
        all_lists.append(not_colored_pieces)
        all_lists.append(colored_pieces)
        all_lists.append(not_solid_pieces)
        all_lists.append(solid_pieces)
        all_lists.append(not_square_pieces)
        all_lists.append(square_pieces)
        
        states = []
        for l in all_lists:
            for i in map(frozenset, combinations(l, 4)):
                states.append(i)
        
        return tuple(states)
    
    def choose_piece(self) -> int:                
        # selecting the piece with the minimum average reward:
        possible_actions = cook_status(self.quarto)['possible_actions']
        reward_dict = dict()
        forbidden_pieces = []
        for a in possible_actions:
            # simulating the state as a list because deepcopy is time_consuming
            board_state = list(self.quarto.get_board_status().ravel())
            spot = a[0]
            board_state[(spot[0] + 4*spot[1])] = a[1]
        #     next_state = tuple(board_state)

        #     if next_state in self.G:
        #         reward = self.G[next_state]
        #     else:
        #         # if the state is not analyzed before, the reward corresponding to the most similar more_crowded state is replaced            
        #         placed_pieces = [p for p in next_state if p != -1]
        #         similar_states = []
        #         for g in self.G:
        #             if not all(item in g for item in placed_pieces):
        #                 break
        #             # similar_states.append((g, similarity))
        #             similarity0 = sum(g[i] == next_state[i] for i in range(16) if g[i] != -1)
        #             similarity1 = -sum(g[i] == -1 for i in range(16))
        #             similar_states.append((g, (similarity0, similarity1)))
        #         fake_state = max(similar_states, key=lambda x: x[1])[0]
        #         reward = self.G[fake_state]
        #     if a[1] in reward_dict:
        #         reward_dict[a[1]].append(reward)
        #     else:
        #         reward_dict[a[1]] = [reward]
        
        # return min(reward_dict.keys(), key=lambda i: np.mean(reward_dict[i]))



            next_state = set_encoder(tuple(board_state))

            for i in next_state:
                if i in self.winning_states:
                    forbidden_pieces.append(a[1])
                    if a[1] in reward_dict:
                        del(reward_dict[a[1]])
                    break

            if a[1] in forbidden_pieces:
                continue

            if next_state in self.G:
                reward = self.G[next_state]
            else:
                # if the state is not analyzed before, the reward corresponding to the most similar more_crowded state is replaced                
                # fake_state = max((g for g in self.G), key=lambda g: (sum(i in g for i in next_state), sum(len(j) for j in g)))
                fake_state = max((g for g in self.G), key=lambda g:
                                (sum(i in g for i in next_state), sum(len(j) for j in g) + sum(len(x) for x in next_state)))

                reward = self.G[fake_state]
            if a[1] in reward_dict:
                reward_dict[a[1]].append(reward)
            else:
                reward_dict[a[1]] = [reward]
        
        if len(reward_dict.keys()) == 0:
            return possible_actions[0][1]
        return min(reward_dict.keys(), key=lambda i: np.mean(reward_dict[i]))



    def place_piece(self) -> tuple[int, int]:   
        # selecting the spot corresponding to the best state
        possible_spots = cook_status(self.quarto)['possible_spots']
        piece = self.quarto.get_selected_piece()
        spot_reward = dict()
        for s in possible_spots:
            # simulating the state as a list because deepcopy is time_consuming
            board_state = list(self.quarto.get_board_status().ravel())
            board_state[(s[0] + 4*s[1])] = piece
        #     next_state = tuple(board_state)

        #     if next_state in self.G:
        #         reward = self.G[next_state]
        #     else:
        #         # if the state is not analyzed before, the reward corresponding to the most similar more_crowded state is replaced
        #         placed_pieces = [p for p in next_state if p != -1]
        #         similar_states = []
        #         for g in self.G:
        #             if not all(item in g for item in placed_pieces):
        #                 break
        #             # similar_states.append((g, similarity))
        #             similarity0 = sum(g[i] == next_state[i] for i in range(16) if g[i] != -1)
        #             similarity1 = -sum(g[i] == -1 for i in range(16))
        #             similar_states.append((g, (similarity0, similarity1)))
        #         fake_state = max(similar_states, key=lambda x: x[1])[0]
        #         # print(next_state)
        #         # print(fake_state)
        #         # input('')
        #         reward = self.G[fake_state]
        #     spot_reward[s] = reward
        # return max(spot_reward.keys(), key=lambda x: spot_reward[x])


            next_state = set_encoder(tuple(board_state))

            for i in next_state:
                if i in self.winning_states:
                    return s

            if next_state in self.G:
                reward = self.G[next_state]
            else:
                # if the state is not analyzed before, the reward corresponding to the most similar more_crowded state is replaced                
                # fake_state = max((g for g in self.G), key=lambda g: (sum(i in g for i in next_state), sum(len(j) for j in g)))
                fake_state = max((g for g in self.G), key=lambda g:
                                (sum(i in g for i in next_state), sum(len(j) for j in g) + sum(len(x) for x in next_state)))
                
                print(next_state)
                print(fake_state)
                input('')

                reward = self.G[fake_state]
            spot_reward[s] = reward
        return max(spot_reward.keys(), key=lambda x: spot_reward[x])



def set_encoder(state: tuple) -> frozenset:
    sety0 = frozenset(state[:4])
    sety1 = frozenset(state[4:8])
    sety2 = frozenset(state[8:12])
    sety3 = frozenset(state[12:16])

    setx0 = frozenset(state[0:16:4])
    setx1 = frozenset(state[1:16:4])
    setx2 = frozenset(state[2:16:4])
    setx3 = frozenset(state[3:16:4])

    setd1 = frozenset((state[0], state[5], state[10], state[15]))
    setd2 = frozenset((state[12], state[9], state[6], state[3]))

    result = frozenset((sety0, sety1, sety2, sety3, setx0, setx1, setx2, setx3, setd1, setd2))
    return result


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
    
    class Agent(object):
        def __init__(self, quarto: Quarto, alpha, random_factor) -> None:
            self.state_history = [(set_encoder(tuple(quarto.get_board_status().ravel())), 0)]  # (initial state, reward)
            self.alpha = alpha
            self.random_factor = random_factor
            self.G = {}
            # initial G with a not-winning full board state
            # self.G[(8, 6, 11, 7, 3, 10, 2, 12, 15, 9, 5, 0, 13, 4, 14, 1)] = -2.0
            # self.G[tuple([-1 for i in range(15)] + [0])] = 0.0
            # self.init_reward()
            self.G[frozenset((frozenset([-1]), frozenset((0, -1))))] = 0.0

        # def init_reward(self):
            # all_pieces = tuple(range(16))
            # len1_sets = [{-1}]
            # len2_sets = list(map(set, product((i for i in all_pieces), [-1])))
            # len3_sets = [i.union({-1}) for i in list(map(set, combinations(all_pieces, 2)))]
            # len4_sets = [i.union({-1}) for i in list(map(set, combinations(all_pieces, 3)))]
            # full_len4_sets = list(map(set, combinations(all_pieces, 4)))
            # all_sets = len1_sets + len2_sets + len3_sets + len4_sets + full_len4_sets
            # final_all_sets = list(map(frozenset, all_sets))
            
            # print(len(final_all_sets))
            # print(final_all_sets[:20])
            # exit()

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

        # def update_state_history(self, state: tuple, reward: float) -> None:
        def update_state_history(self, state: frozenset, reward: float) -> None:
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
    win = 0

    for i in range(1000):
            quarto_copy = deepcopy(quarto)
            RL_player = RLPlayer(quarto_copy, agent.G)
            random_player = RandomPlayer(quarto_copy)
            agent_state = None
            
            # the starting player changes each turn            
            if i % 2 == 0:            
                while not quarto_copy.check_finished():
                    selected_piece = agent.choose_piece(cook_status(quarto_copy)['possible_pieces'], RL_player)
                    quarto_copy.select(selected_piece)

                    select_ok = False
                    while not select_ok:
                        select_ok = quarto_copy.place(*random_player.place_piece())
                    
                    if not quarto_copy.check_winner():
                        # give a -25 reward to the agent state if it loses
                        agent.update_state_history(agent_state, -10.0)
                        break
                    
                    elif quarto_copy.check_finished():
                        # give a -5 reward to the agent state if it is a tie
                        agent.update_state_history(agent_state, -2.0)
                        break

                    elif agent_state is not None:
                        # give a -2 reward to the agent state if the game is not finished
                        agent.update_state_history(agent_state, 0.0)               
                    
                    select_ok = False
                    while not select_ok:
                        select_ok = quarto_copy.select(random_player.choose_piece())
                    
                    selected_spot = agent.place_piece(cook_status(quarto_copy)['possible_spots'], RL_player)
                    quarto_copy.place(*selected_spot)
                    
                    # agent_state = tuple(quarto_copy.get_board_status().ravel())                
                    agent_state = set_encoder(tuple(quarto_copy.get_board_status().ravel()))
                    if not quarto_copy.check_winner():
                        # give a 5 reward to the agent state if it wins
                        agent.update_state_history(agent_state, 5.0)
                        win += 1
                        break

                    elif quarto_copy.check_finished():
                        # give a -5 reward to the agent state if it is a tie
                        agent.update_state_history(agent_state, -2.0)
                        break

            else:
                while not quarto_copy.check_finished():
                    select_ok = False
                    while not select_ok:
                        select_ok = quarto_copy.select(random_player.choose_piece())
                    
                    selected_spot = agent.place_piece(cook_status(quarto_copy)['possible_spots'], RL_player)
                    quarto_copy.place(*selected_spot)
                    
                    # agent_state = tuple(quarto_copy.get_board_status().ravel())                
                    agent_state = set_encoder(tuple(quarto_copy.get_board_status().ravel()))                
                    if not quarto_copy.check_winner():
                        # give a 5 reward to the agent state if it wins
                        agent.update_state_history(agent_state, 5.0)
                        win += 1
                        break

                    elif quarto_copy.check_finished():
                        # give a -5 reward to the agent state if it is a tie
                        agent.update_state_history(agent_state, -2.0)
                        break
                    
                    selected_piece = agent.choose_piece(cook_status(quarto_copy)['possible_pieces'], RL_player)
                    quarto_copy.select(selected_piece)

                    select_ok = False
                    while not select_ok:
                        select_ok = quarto_copy.place(*random_player.place_piece())
                    
                    if not quarto_copy.check_winner():
                        # give a -25 reward to the agent state if it loses
                        agent.update_state_history(agent_state, -10.0)
                        break
                    
                    elif quarto_copy.check_finished():
                        # give a -5 reward to the agent state if it is a tie
                        agent.update_state_history(agent_state, -2.0)
                        break

                    elif agent_state is not None:
                        # give a -2 reward to the agent state if the game is not finished
                        agent.update_state_history(agent_state, 0.0)
                    
            agent.learn()
            if i % 50 == 0:
                print(f'{i}: {win/10}')
    
    return agent.G