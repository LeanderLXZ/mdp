import numpy as np


size = 500
n = 100

board = np.array([np.nan] * size * size)
board[np.random.choice(size * size, 1 * n)] = 10000
board[np.random.choice(size * size, 4 * n)] = 1000
board[np.random.choice(size * size, 20 * n)] = 0
board[np.random.choice(size * size, 10 * n)] = -100
board[np.random.choice(size * size, 4 * n)] = -1000
board[np.random.choice(size * size, 2 * n)] = -10000

board = board.reshape((size, size)).tolist()
for i in range(size):
    for j in range(size):
        board[i][j] = str(board[i][j])
        if board[i][j] == 'nan':
            board[i][j] = 'X'

with open('../inputs/i8.txt', 'w') as f:
    f.write('{}\n'.format(size))
    f.write('0.9\n')
    f.write('0.8, 0.1, 0.1\n')
    f.write('\n')
    for row in board:
        f.write(','.join(row) + '\n')
