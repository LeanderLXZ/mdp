import numpy as np
import pandas as pd
from time import time
from copy import deepcopy

from utils import *
from value_iteration import ValueIteration, IterationConverged


pd.set_option('precision', 2)


class PolicyIteration(ValueIteration):
    
    def __init__(self, 
                 board_file_path, 
                 threshold=0.01,
                 init_policy_direction=None,
                 improve_p_with_v=False,
                 use_arrow=False,
                 verbose=False):
        super(PolicyIteration, self).__init__(board_file_path, 
                         threshold=threshold, 
                         use_arrow=use_arrow, 
                         verbose=verbose)
        
        # Initialize the policy
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i, j) in self.non_terminal_states:
                    if init_policy_direction:
                        self.policy[i][j] = \
                            self.direction_str[init_policy_direction]
                    else:
                        self.policy[i][j] = \
                            self.direction_str[np.random.choice(4)]
                else:
                    self.policy[i][j] = self.values[i][j]

        self.improve_p_with_v = improve_p_with_v
        
    
    def get_directions(self, policy_direction):
        
        if policy_direction == self.direction_str[0]:
            return  [(-1, 0), (0, 1), (1, 0), (0, -1)]
        elif policy_direction == self.direction_str[1]:
            return  [(0, 1), (1, 0), (0, -1), (-1, 0)]
        elif policy_direction == self.direction_str[2]:
            return  [(1, 0), (0, -1), (-1, 0), (0, 1)]
        elif policy_direction == self.direction_str[3]:
            return  [(0, -1), (-1, 0), (0, 1), (1, 0)]
        else:
            raise ValueError(policy_direction)
    
    # @print_time
    def evaluate_policy(self):
        
        new_values = deepcopy(self.values)
        
        delta = 0
        for (i, j) in self.non_terminal_states:
            directions = self.get_directions(self.policy[i][j])
            q_value = self.calc_q_value(i, j, directions, self.noises)
            delta = max(delta, np.abs(q_value - self.values[i, j]))
            new_values[i, j] = q_value
       
        self.values = new_values 
        if delta < self.threshold:
            raise IterationConverged
    
    # @print_time
    def improve_policy(self):
        
        new_policy = deepcopy(self.policy)
        
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        converged = True
        
        for i, j in self.non_terminal_states:
            q_values = []
            for _ in range(4):
                q_values.append(
                    self.calc_q_value(i, j, directions, self.noises))
                directions.append(directions.pop(0))
            p = self.direction_str[np.argmax(q_values)]
            if p != new_policy[i][j]:
                converged = False
                new_policy[i][j] = p
        
        if converged:
            raise IterationConverged
        else:
            self.policy = new_policy
    
    def improve_policy_with_value(self):
        
        new_policy = deepcopy(self.policy)
        new_values = deepcopy(self.values)
        
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        converged = True
        
        for i, j in self.non_terminal_states:
            q_values = []
            for _ in range(4):
                q_values.append(
                    self.calc_q_value(i, j, directions, self.noises))
                q_values = [round(v, 6) for v in q_values]
                directions.append(directions.pop(0))
            p = self.direction_str[np.argmax(q_values)]
            if p != new_policy[i][j]:
                converged = False
                new_policy[i][j] = p
            new_values[i, j] = max(q_values)
        
        self.values = new_values
        if converged:
            raise IterationConverged
        else:
            self.policy = new_policy
        
    def run(self):
        
        print('Initial policy:')
        print(pd.DataFrame(self.policy))
        
        strat_time = time.time()
        count = 0
        i = 0 
        
        while True:
            if self.verbose:
                print('-' * 60)
                print('Policy iteration:', i)
            
            try:
                j = 0
                while True:
                    if self.verbose:
                        print('Value iteration:', j)
                    self.evaluate_policy()
                    j += 1
                    count += 1
            except IterationConverged:
                pass
            
            try:
                if self.improve_p_with_v:
                    self.improve_policy_with_value()
                else:
                    self.improve_policy()
                # if self.verbose:
                #     print('Values:')
                #     print(pd.DataFrame(self.values))
                #     print('Policy:')
                #     print(pd.DataFrame(self.policy))
            except IterationConverged:
                run_time = time.time() - strat_time
                print('-' * 60)
                print('Final values:')
                print(pd.DataFrame(self.values))
                print('Final policy:')
                print(pd.DataFrame(self.policy))
                print('Total value iterations:', count)
                print('Total policy iterations:', i)
                print('Runtime:{}s'.format(run_time))
                break
            
            i += 1
            
        return self.board_size, count, i, run_time

            
if __name__ == '__main__':

    _ = PolicyIteration('../inputs/i7.txt', threshold=0.01, 
                        init_policy_direction=1, improve_p_with_v=True,
                        use_arrow=True, verbose=True).run()
