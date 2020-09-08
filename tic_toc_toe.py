"""
Reinforcement Learning for Tic-Toc-Toe Game

Agent & Training:
1. The agent is trained by a temporal difference method by feeding reward back to the state of the board.
2. An explore rate is set so that the agent will explore new moves ignoring the value of the state.
3. If the state has never been seen by the agent, it is set to neutral (0.5).
4. By setting the explore rate to 0.3, and after training for 1000 rounds (both first turn and second turn), the agent can almost beat human players.

Algorithm
The state of the board is represented by 0 (empty), 1 (current player) and 2 (other player). It is further mapped to a 3-based integer to be saved for value evaluation.

A few more things can be done:
1. how to evaluate the level of the agent
2. how to make use of the symmetry of the state

author: zl
date: 2020/09/06
"""

#%%
from os import truncate
import random
import numpy
from numpy.__config__ import show
import pickle
from itertools import cycle

class Board():

    def __init__(self):
        self.state = list(9*' ')

    def __str__(self):
        return self.state

    def get_valid_position(self):
        pos = [i for i, s in enumerate(self.state) if s==' ']
        return pos

    def show(self):
        display = '''
         {} | {} | {}
        -----------
         {} | {} | {}
        -----------
         {} | {} | {}'''
        print(display.format(*self.state))


def pos2grid(pos):
    return (pos//3, pos%3)


def state2repr(state, mark):
    repr = [1 if c==mark else 0 if c==' ' else 2 for c in state]
    return repr


def repr2hash(repr):
    """
    repr = representation of board state, 0 for empty, 1 for current player, 2 for the other player
    """
    # transform state to a 3-based number
    hash = 0
    for i in repr:
        hash = hash*3 + i
    return hash


class Player():
    def __init__(self, name, mark):
        self.name = name
        self.mark = mark

    def generate_move(self, board):
        valid_positions = board.get_valid_position()
        pos = random.choice(valid_positions)
        return pos


class HumanPlayer(Player):
    def generate_move(self, board):
        valid_positions = board.get_valid_position()
        
        print('\n Available positions: {}'.format(valid_positions))

        pos = -1
        while pos not in valid_positions:
            pos = int(input('Please choose your next move'))

        return pos


class AgentPlayer(Player):
    def __init__(self, name, mark, stepsize=0.1, explorerate=0, show_value=False):
        self.name = name
        self.mark = mark
        self.value = dict()
        self.stepsize = stepsize
        self.explorerate = explorerate
        self.show_value = show_value

    def generate_move(self, board):
        valid_positions = board.get_valid_position()
        # transform state to a consistent representation, 1 means current player
        repr = state2repr(board.state, self.mark)

        is_explore = random.random() < self.explorerate
        if is_explore:
            # naive random policy
            best_pos = random.choice(valid_positions)

            nextrepr = repr.copy()
            nextrepr[best_pos] = 1
            hashval = repr2hash(nextrepr)
            if hashval not in self.value:
                self.value[hashval] = 0.5
        
        else:
            # choose the position of best value
            best_pos = 0
            best_value = 0
            values = []

            # shuffle positions to make sure random choice among same valued positions
            random.shuffle(valid_positions)

            for pos in valid_positions:
                nextrepr = repr.copy() # list copy
                nextrepr[pos] = 1
                hashval = repr2hash(nextrepr)
                if hashval in self.value:
                    value = self.value[hashval]
                else:
                    value = 0.5
                    self.value[hashval] = value
                values.append(value)

                if value > best_value:
                    best_pos = pos
                    best_value = value

            if self.show_value:
                display = sorted(dict(zip(valid_positions, values)).items())
                print(display)

        return best_pos

    def update_policy(self, hashvals, reward):
        target = reward
        for hashval in reversed(hashvals):
            self.value[hashval] = self.value[hashval] + self.stepsize * (target - self.value[hashval])
            target = self.value[hashval]

    def load_policy(self, path):
        with open(path, 'rb') as f:
            self.value = pickle.load(f)

    def save_policy(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.value, f)


def is_winning_move(board):
    wins = ((0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6))
    
    for a,b,c in wins:
        chars = board.state[a] + board.state[b] + board.state[c]
        if len(set(chars)) == 1 and chars != 3*' ':
            return True
    return False


def play(players, show_game=False):
    # new game
    board = Board()
    history = {p.name:[] for p in players}

    # one game
    for i, p in zip(range(9), cycle(players)):
        pos = p.generate_move(board)
        board.state[pos] = p.mark
        hashval = repr2hash(state2repr(board.state, p.mark))
        history[p.name].append(hashval)

        if show_game:
            print('\n {}({}) moves {}'.format(p.name, p.mark, pos2grid(pos)))
            board.show()

        # check if game end
        if is_winning_move(board):
            print('{} Wins !'.format(p.name))
            return p.name, history

    print('Draw !')
    return 'draw', history


#%% train agent players

def train(players, round=100, show_game=False, policy_path=None):
    count = [0, 0, 0] # win, draw, lose

    for i in range(1, round+1):
        print('Game {}.'.format(i), end=' ')

        # start a new game
        result, history = play(players, show_game=show_game)
        reward = [1 if result==p.name else 0.5 if result=='draw' else 0 for p in players]

        i = 0 if result==p1.name else 1 if result=='draw' else 2
        count[i] += 1
        print('({})'.format(count))

        # feed reward
        for p, r in zip(players, reward):
            p.update_policy(history[p.name], r)

    # save policy
    if policy_path is not None:
        for p, path in zip(players, policy_path):
            p.save_policy(path)

# two trained agents
p1 = AgentPlayer('p1', 'o', explorerate=0.3)
p2 = AgentPlayer('p2', 'x', explorerate=0.3)
players = [p1, p2]
policy_path = [p.name+'_er='+str(p.explorerate)+'.pkl' for p in players]

# p1 first
train(players, round=1000, show_game=False, policy_path=policy_path)
# p2 first
train(list(reversed(players)), round=1000, show_game=False, policy_path=policy_path)

# one trained, one random
# p1 = AgentPlayer('p1', 'o', policy='trained')
# p2 = AgentPlayer('p2', 'x', policy='random')
# players = [p1, p2]
# train(players, round=1000, show_game=False, policy_path=['p1_1train'])

#%% test
train(list(reversed(players)), round=1, show_game=True)

#%% play with trained agents, agent first
p1 = AgentPlayer('p1', 'o', explorerate=0, show_value=True)
p1.load_policy('p1_er=0.3.pkl')
# p2 = AgentPlayer('p2', 'x', policy='trained', show_value=True)
# p2.load_policy('p1_1train.pkl')
p2 = HumanPlayer('p2', 'x')

players = [p1, p2]
play(players, show_game=True)

#%% human first
p1 = HumanPlayer('p1', 'o')

p2 = AgentPlayer('p2', 'x', explorerate=0, show_value=True)
p2.load_policy('p1_er=0.3.pkl')

players = [p1, p2]
play(players, show_game=True)
