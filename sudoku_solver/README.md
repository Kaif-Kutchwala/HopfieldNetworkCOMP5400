# Hopfield Networks: Generation and Solving of Sudokus

This repository contains an implementation of a stochastic Hopfield Network with Simulated Annealing, and a configurable script that can be used to solve a specified number of puzzles. The difficulty of the puzzle can be changed by increasing or decreasing the number of cells set to zero during the generation of the puzzle.

```
Before running any script make sure all the dependencies are installed.
Errors should tell you which dependencies are missing.
```

# File:


The following files are provided:

| File | Purpose |
| ---  | --- |
| `solve_sudoku.py`      | Python Implementation of Sudoku Solver|


# Usage: `solve_sudoku.py`

To run this application simply execute the following in your terminal:
```
python solve_sudoku.py
```

This should display the puzzles generated and then the solutions to the respective puzzles in the console window.

### Controls

```
Set the 'num_of_zeros'at the top of the file to the number cells in the 4 by 4 sudoku puzzle to set to '0'
```
```
Set the 'num_of_puzzles' at the top of the file the desired number of puzzles wanted
```
### Note
Other unsuccessful attempts at building the solver can be found in the repository linked in the Appendix of the report submited or by going to the link below.

GitHub - Kaif-Kutchwala/HopfieldNetworkCOMP5400

