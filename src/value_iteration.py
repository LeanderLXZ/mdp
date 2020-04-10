import numpy as np
import pandas as pd
from time import time
from copy import deepcopy

from utils import *


pd.set_option('precision', 2) 


class IterationConverged(Exception):
    pass


class ValueIteration(object):
    
    
    def __init__(self, 
                 board_file_path, 
                 threshold=0.01, 
                 use_arrow=False, 
                 verbose=False):
        self.board_file_path = board_file_path 
        self.board_size = 1
        self.gamma = 1
        self.noises = []
        self.non_terminal_states = []
        self.values = self.get_input_values()
        self.threshold = threshold
        self.verbose = verbose
        
        # Policy
        self.policy = []
        for _ in range(self.board_size):
            self.policy.append([None] * self.board_size)
        
        if use_arrow:
            self.direction_str = ['↑', '→', '↓', '←']
        else:
            self.direction_str = ['up', 'right', 'down', 'left']
        
    def load_file(self):
        lines = []
        with open(self.board_file_path, 'r') as f:
            for i, line in enumerate(f):
                c = line.split('#')[0].replace(
                    ' ', '').replace('\n', '').split(',')
                if c == ['']:
                    continue
                else:
                    lines.append(c)
        return lines
        
    def get_input_values(self):
        
        l = self.load_file()
        
        self.board_size = int(l[0][0])
        self.gamma = float(l[1][0])
        if len(l[2]) == 3:
            noises = [l[2][0], l[2][1], 0., l[2][2]]
        else:
            noises = [l[2][0], l[2][1], l[2][3], l[2][2]]
        self.noises = [float(n) for n in noises]
        
        values = np.zeros((self.board_size, self.board_size))
        for i in range(self.board_size):
            for j in range(self.board_size):
                v = l[3 + i][j]
                if v == 'X':
                    self.non_terminal_states.append((i, j))
                    values[i, j] = 0.
                else:
                    values[i, j] = float(v)
        
        print('-' * 60)
        print('Board size:', self.board_size)
        print('Gamma: ', self.gamma)
        print('Noise (clockwise):', self.noises)
        print('Initial board states:')
        print(pd.DataFrame(l[3: 3 + self.board_size]))
        
        return values
    
    def get_final_policy(self):
        
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for i, row in enumerate(self.values):
            for j, value in enumerate(row):
                if (i, j) in self.non_terminal_states:
                    q_values = []
                    for _ in range(4):
                        q_values.append(
                            self.calc_q_value(i, j, directions, self.noises))
                        directions.append(directions.pop(0)) 
                    assert directions == [(-1, 0), (0, 1), (1, 0), (0, -1)]
                    self.policy[i][j] = self.direction_str[np.argmax(q_values)]
                else:
                    self.policy[i][j] = self.values[i][j]
        
    def calc_q_value(self, i, j, directions, noises):
        v = 0.
        for (d_i, d_j), noise in zip(directions, noises):
            i_next, j_next = i + d_i, j + d_j
            if 0 <= i_next < self.board_size and 0 <= j_next < self.board_size:
                v_next = self.values[i_next, j_next]
            else:
                v_next = self.values[i, j]
            v += noise * self.gamma * v_next
        return v
    
    # @print_time
    def update_values(self):
        
        new_values = deepcopy(self.values)
        
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        delta = 0
        for (i, j) in self.non_terminal_states:
            q_values = []
            for _ in range(4):
                q_values.append(
                    self.calc_q_value(i, j, directions, self.noises))
                directions.append(directions.pop(0)) 
            best_value = max(q_values)
            delta = max(delta, np.abs(best_value - self.values[i, j]))
            new_values[i, j] = best_value
        
        self.values = new_values
        if delta < self.threshold:
            raise IterationConverged

    def run(self):
        
        strat_time = time.time()
        
        try:
            i = 0
            while True:
                self.update_values()
                if self.verbose:
                    print('-' * 60)
                    print('Iteration:', i)
                    # print('Values:')
                    # print(pd.DataFrame(self.values))
                i += 1
        except IterationConverged:
            run_time = time.time() - strat_time
            self.get_final_policy()
            print('-' * 60)
            print('Final values:')
            print(pd.DataFrame(self.values))
            print('Final policy:')
            print(pd.DataFrame(self.policy))
            print('Total value iterations:', i)
            print('Runtime:{}s'.format(run_time))

        return self.board_size, i, run_time
                        
            
if __name__ == '__main__':

    _ = ValueIteration(
        '../input/i7.txt',
        threshold=0.01,
        use_arrow=True,
        verbose=False
    ).run()
