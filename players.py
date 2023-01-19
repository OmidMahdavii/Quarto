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


class SemiRandomPlayer(Player):
    """
    Performs a winning move if there is any
    """

    def __init__(self, quarto: Quarto) -> None:
        super().__init__(quarto)
        self.quarto = quarto


    def choose_piece(self) -> int:
        cooked = cook_status(self.quarto)
        possible_actions = cooked['possible_actions']
        possible_pieces = set(cooked['possible_pieces'])
        forbidden_pieces = set()
        for a in possible_actions:
            if a[1] in forbidden_pieces:
                continue
            
            # simulating the state as a list because deepcopy is time_consuming
            board_state = list(self.quarto.get_board_status().ravel())
            spot = a[0]
            board_state[(spot[0] + 4*spot[1])] = a[1]

            next_state = state_encoder(tuple(board_state))

            # if the next state is a winning one don't choose the corresponding piece
            for i in next_state:
                if i[1] == 4:
                    forbidden_pieces.add(a[1])
                    break

        not_forbidden_pieces = possible_pieces.difference(forbidden_pieces)
        if len(not_forbidden_pieces) != 0:
            return np.random.choice(list(not_forbidden_pieces))
        
        return np.random.choice(list(possible_pieces))
        
    def place_piece(self) -> tuple[int, int]:
        possible_spots = cook_status(self.quarto)['possible_spots']
        piece = self.quarto.get_selected_piece()
        for s in possible_spots:
            # simulating the state as a list because deepcopy is time_consuming
            board_state = list(self.quarto.get_board_status().ravel())
            board_state[(s[0] + 4*s[1])] = piece
            next_state = state_encoder(tuple(board_state))

            # if the next state is a winning one return the chosen spot
            for i in next_state:
                if i[1] == 4:
                    return s

        return possible_spots[np.random.choice(len(possible_spots))]


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
        forbidden_pieces = []
        for a in possible_actions:
            # simulating the state as a list because deepcopy is time_consuming

            if a[1] in forbidden_pieces:
                continue

            board_state = list(self.quarto.get_board_status().ravel())
            spot = a[0]
            board_state[(spot[0] + 4*spot[1])] = a[1]

            next_state = state_encoder(tuple(board_state))

            # if the next state is a winning one don't ever choose the corresponding piece
            for i in next_state:
                if i[1] == 4:
                    forbidden_pieces.append(a[1])
                    if a[1] in reward_dict:
                        del(reward_dict[a[1]])
                    break

            if a[1] in forbidden_pieces:
                continue

            reward = self.G[next_state]
            
            if a[1] in reward_dict:
                reward_dict[a[1]].append(reward)
            else:
                reward_dict[a[1]] = [reward]
        
        # if there is at least one winning state for every piece select the piece with the minimum average reward
        if len(reward_dict.keys()) == 0:
            for a in possible_actions:
                board_state = list(self.quarto.get_board_status().ravel())
                spot = a[0]
                board_state[(spot[0] + 4*spot[1])] = a[1]

                next_state = state_encoder(tuple(board_state))

                reward = self.G[next_state]

                if a[1] in reward_dict:
                    reward_dict[a[1]].append(reward)
                else:
                    reward_dict[a[1]] = [reward]
        
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
            next_state = state_encoder(tuple(board_state))
            
            # if the next state is a winning one return the chosen spot
            for i in next_state:
                if i[1] == 4:
                    return s

            spot_reward[s] = self.G[next_state]
        
        return max(spot_reward.keys(), key=lambda x: spot_reward[x])



def state_encoder(state: tuple) -> tuple:
    not_high = list(range(8))
    high = list(range(8, 16))
    not_colored = list(range(4)) + list(range(8, 12))
    colored = list(range(4, 8)) + list(range(12, 16))
    not_solid = [0, 1, 4, 5, 8, 9, 12, 13]
    solid = [2, 3, 6, 7, 10, 11, 14, 15]
    not_square = list(range(0, 16, 2))
    square = list(range(1, 16, 2))
    
    matches = (not_high, high, not_colored, colored, not_solid, solid, not_square, square)
    
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

    state = set((sety0, sety1, sety2, sety3, setx0, setx1, setx2, setx3, setd1, setd2))
    
    stat = set() # it is going to be a set containing 10 tuples. each tuple contains 8 inner tuples. each inner tuple corresponds
                 # to a specific characteristic and the number of pieces sharing the characteristic in a row, column or diameter

    for series in state:
        stat.add(tuple((tuple((idx, sum((i in series) for i in match)))) for idx, match in enumerate(matches)))
    
    map_state = []
    for idx in range(8):
        max = 0
        for s in stat:
            if s[idx][1] > max:
                max = s[idx][1]
        map_state.append((idx, max))

    return tuple(map_state)


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


def learning(alpha=0.15, random_factor=0.2, gamma=1.0) -> dict:
    """
    reinforcement learning process used to calculate the policy
    """
    
    class Agent(object):
        def __init__(self, quarto: Quarto, alpha, random_factor, gamma) -> None:
            self.state_history = [(state_encoder(tuple(quarto.get_board_status().ravel())), 0)]  # (initial state, reward)
            self.alpha = alpha
            self.random_factor = random_factor
            self.gamma = gamma
            self.G = {}
            self.init_reward()

        def init_reward(self):
            match0 = product([0], range(5))
            match1 = product([1], range(5))
            match2 = product([2], range(5))
            match3 = product([3], range(5))
            match4 = product([4], range(5))
            match5 = product([5], range(5))
            match6 = product([6], range(5))
            match7 = product([7], range(5))
            for i in product(match0, match1, match2, match3, match4, match5, match6, match7):
                self.G[i] = np.random.uniform(low=1.0, high=0.1)

        def place_piece(self, possible_spots: tuple, player: RLPlayer) -> tuple[int, int]:
            # selected_spot = None
            randomN = np.random.random()
            if randomN < self.random_factor:
                # select a random spot
                selected_spot = possible_spots[np.random.choice(len(possible_spots))]
            else:
                # if exploiting, gather all possible actions and choose one with the highest G (reward)
                selected_spot = player.place_piece()
            return selected_spot

        def choose_piece(self, possible_pieces: tuple, player: RLPlayer) -> int:
            # selected_piece = None
            randomN = np.random.random()
            if randomN < self.random_factor:
                # select a random piece
                selected_piece = np.random.choice(list(possible_pieces))
            else:
                # if exploiting, gather all possible pieces and choose one with the lowest average reward
                selected_piece = player.choose_piece()
            return selected_piece

        def update_state_history(self, state: tuple, reward: float) -> None:
            self.state_history.append((state, reward))

        def learn(self) -> None:
            target = 0.0

            for prev, reward in reversed(self.state_history):
                target += reward
                self.G[prev] = self.G[prev] + self.alpha * (target - self.G[prev])
                target *= self.gamma

            self.state_history = []

            self.random_factor -= 10e-5  # decrease random factor each episode of play

    quarto = Quarto()
    agent = Agent(quarto, alpha, random_factor, gamma)
    win = 0

    for i in range(10000):
            quarto_copy = deepcopy(quarto)
            RL_player = RLPlayer(quarto_copy, agent.G)
            # random_player = RandomPlayer(quarto_copy)
            random_player = SemiRandomPlayer(quarto_copy)
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
                        # give a -5 reward to the agent state if it loses
                        agent.update_state_history(agent_state, -5.0)
                        break
                    
                    elif quarto_copy.check_finished():
                        # give a -2 reward to the agent state if it is a tie
                        agent.update_state_history(agent_state, -2.0)
                        break

                    elif agent_state is not None:
                        # give a -0 reward to the agent state if the game is not finished
                        agent.update_state_history(agent_state, 0.0)               
                    
                    select_ok = False
                    while not select_ok:
                        select_ok = quarto_copy.select(random_player.choose_piece())
                    
                    selected_spot = agent.place_piece(cook_status(quarto_copy)['possible_spots'], RL_player)
                    quarto_copy.place(*selected_spot)
                    
                    agent_state = state_encoder(tuple(quarto_copy.get_board_status().ravel()))
                    if not quarto_copy.check_winner():
                        # give a 5 reward to the agent state if it wins
                        agent.update_state_history(agent_state, 5.0)
                        win += 1
                        break

                    elif quarto_copy.check_finished():
                        # give a -2 reward to the agent state if it is a tie
                        agent.update_state_history(agent_state, -2.0)
                        break

            else:
                while not quarto_copy.check_finished():
                    select_ok = False
                    while not select_ok:
                        select_ok = quarto_copy.select(random_player.choose_piece())
                    
                    selected_spot = agent.place_piece(cook_status(quarto_copy)['possible_spots'], RL_player)
                    quarto_copy.place(*selected_spot)
                    
                    agent_state = state_encoder(tuple(quarto_copy.get_board_status().ravel()))                
                    if not quarto_copy.check_winner():
                        # give a 5 reward to the agent state if it wins
                        agent.update_state_history(agent_state, 5.0)
                        win += 1
                        break

                    elif quarto_copy.check_finished():
                        # give a -2 reward to the agent state if it is a tie
                        agent.update_state_history(agent_state, -2.0)
                        break
                    
                    selected_piece = agent.choose_piece(cook_status(quarto_copy)['possible_pieces'], RL_player)
                    quarto_copy.select(selected_piece)

                    select_ok = False
                    while not select_ok:
                        select_ok = quarto_copy.place(*random_player.place_piece())
                    
                    if not quarto_copy.check_winner():
                        # give a -5 reward to the agent state if it loses
                        agent.update_state_history(agent_state, -5.0)
                        break
                    
                    elif quarto_copy.check_finished():
                        # give a -2 reward to the agent state if it is a tie
                        agent.update_state_history(agent_state, -2.0)
                        break

                    elif agent_state is not None:
                        # give a -0 reward to the agent state if the game is not finished
                        agent.update_state_history(agent_state, 0.0)
                    
            agent.learn()
            if i % 200 == 0:
                f = open(f'record/percentage_alpha{round(alpha*100)}_random{round(random_factor*100)}_gamma{round(gamma*100)}.txt','a')
                f.write(f'{i}: {win/2.0}%\r')
                f.close()
                win = 0
    
    return agent.G