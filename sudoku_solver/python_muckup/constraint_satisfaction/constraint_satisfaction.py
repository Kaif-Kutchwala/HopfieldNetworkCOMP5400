# Define a function to check if a given value is valid in the current position on the board
def is_valid(row, col, val, board):
    # Check if the value is already in the same row or column
    for i in range(9):
        if board[row][i] == val or board[i][col] == val:
            return False

    # Check if the value is already in the same 3x3 box
    for i in range(3):
        for j in range(3):
            if board[3 * (row // 3) + i][3 * (col // 3) + j] == val:
                return False

    # If the value is not in the same row, column, or 3x3 box, it is valid
    return True

# Define a function to solve the Sudoku puzzle using backtracking


def solve(board):
    # Loop through each row and column on the board
    for row in range(9):
        for col in range(9):
            # Check if the current position is empty (represented by 0)
            if board[row][col] == 0:
                # Try each possible value (1-9) in the current position
                for val in range(1, 10):
                    # Check if the value is valid in the current position
                    if is_valid(row, col, val, board):
                        # If the value is valid, set it in the current position on the board
                        board[row][col] = val

                        # Recursively solve the remaining board using the updated position
                        if solve(board):
                            # If a solution is found, return True
                            return True
                        else:
                            # If a solution is not found, reset the current position to 0 and try the next value
                            board[row][col] = 0

                # If no value is valid in the current position, backtrack to the previous position
                return False

    # If all positions on the board have been filled with valid values, the puzzle is solved
    return True

# Define a function to print the board in a more readable format


def print_board(board):
    for i in range(9):
        for j in range(9):
            print(board[i][j], end=' ')
        print()


# Define the starting board as a 2D array
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
