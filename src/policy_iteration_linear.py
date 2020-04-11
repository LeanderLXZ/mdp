import numpy as np
import pandas as pd
from time import time
from copy import deepcopy

from utils import *
from value_iteration import IterationConverged
from policy_iteration import PolicyIteration


pd.set_option('precision', 2)


class PolicyIterationLinear(PolicyIteration):
    
    def __init__(self, 
                 board_file_path, 
                 threshold=0.01, 
                 init_policy_direction=None, 
                 improve_p_with_v=False, 
                 use_arrow=False, 
                 verbose=False):
        super(PolicyIterationLinear, self).__init__(
            board_file_path, 
            threshold=threshold, 
            init_policy_direction=init_policy_direction,
            improve_p_with_v=improve_p_with_v, 
            use_arrow=use_arrow, 
            verbose=verbose
            )

    # @print_time
    def evaluate_policy(self):
        
        A = np.zeros((self.board_size ** 2, self.board_size ** 2))
        b = np.zeros(self.board_size ** 2)
        i_A = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                A[i_A][i * self.board_size + j] = 1
                if (i, j) in self.non_terminal_states:
                    for iter_d, (d_i, d_j) in enumerate(
                            self.get_directions(self.policy[i][j])):
                        i_next, j_next = i + d_i, j + d_j
                        if 0 <= i_next < self.board_size and \
                                0 <= j_next < self.board_size:
                            idx = i_next * self.board_size + j_next
                            A[i_A][idx] = - self.noises[iter_d] * self.gamma
                        else:
                            A[i_A][i * self.board_size + j] -= \
                                self.noises[iter_d] * self.gamma
                else:
                    b[i_A] = self.values[i, j]
                i_A += 1
        
        new_values = np.linalg.solve(A, b).reshape(
            (self.board_size, self.board_size))

        self.values = new_values

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
                q_values = [round(v, 6) for v in q_values]
                directions.append(directions.pop(0))
            p = self.direction_str[np.argmax(q_values)]
            if p != self.policy[i][j]:
                converged = False
                # print(i, j)
                # print(q_values)
                # print(np.argmax(q_values))
                new_policy[i][j] = p
        
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
            
            print('Solving linear equations...')
            self.evaluate_policy()
            
            try:
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

    _ = PolicyIterationLinear('../inputs/i3.txt', threshold=0.01,
                              init_policy_direction=1, improve_p_with_v=True,
                              use_arrow=True, verbose=True).run()
