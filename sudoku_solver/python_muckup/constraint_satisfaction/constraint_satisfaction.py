def is_valid(row, col, val, board):
    for i in range(9):
        if board[row][i] == val:
            return False
        if board[i][col] == val:
            return False
        if board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == val:
            return False
    return True

def solve(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for val in range(1, 10):
                    if is_valid(row, col, val, board):
                        board[row][col] = val
                        if solve(board):
                            return True
                        else:
                            board[row][col] = 0
                return False
    return True

def print_board(board):
    for i in range(9):
        for j in range(9):
            print(board[i][j], end=' ')
        print()

board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

if solve(board):
    print_board(board)
else:
    print("No solution found")
