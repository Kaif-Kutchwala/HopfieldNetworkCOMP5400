#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <cassert>
#include <string>

using namespace std;

class HopfieldNetwork {
private:
    int num_neurons;
    vector<vector<double>> weights;

public:
    HopfieldNetwork(int n) {
        num_neurons = n;
        weights = vector<vector<double>>(n, vector<double>(n, 0));
    }

    void train(vector<vector<double>> patterns) {
        for (int i = 0; i < num_neurons; i++) {
            for (int j = 0; j < num_neurons; j++) {
                if (i == j) {
                    weights[i][j] = 0;
                }
                else {
                    double sum = 0;
                    for (auto pattern : patterns) {
                        sum += pattern[i] * pattern[j];
                    }
                    weights[i][j] = sum / num_neurons;
                }
            }
        }
    }

    vector<double> recall(vector<double> input) {
        vector<double> prev_output = input;
        vector<double> new_output = prev_output;
        while (true) {
            for (int i = 0; i < num_neurons; i++) {
                double activation = 0;
                for (int j = 0; j < num_neurons; j++) {
                    activation += weights[i][j] * prev_output[j];
                }
                new_output[i] = activation;
            }
            if (prev_output == new_output) {
                break;
            }
            prev_output = new_output;
        }
        return new_output;
    }
};

//vector<double> sudoku_to_vector(string puzzle) {
//    vector<double> vec(81 * 10, 0);
//    for (int i = 0; i < 81; i++) {
//        if (puzzle[i] != '.') {
//            int digit = puzzle[i] - '0';
//            vec[i * 10 + digit - 1] = 1;
//        }
//    }
//    return vec;
//}

string vector_to_sudoku(vector<double> vec) {
    string puzzle = "";
    for (int i = 0; i < 81; i++) {
        int max_index = -1;
        double max_value = -INFINITY;
        for (int j = 0; j < 10; j++) {
            double value = vec[i * 10 + j];
            if (value > max_value) {
                max_index = j;
                max_value = value;
            }
        }
        if (max_index == -1) {
            puzzle += ".";
        }
        else {
            puzzle += to_string(max_index + 1);
        }
        if ((i + 1) % 9 == 0) {
            puzzle += "\n";
        }
    }
    return puzzle;
}

vector<double> sudoku_to_vector(string puzzle) {
    vector<double> vec(81 * 10, 0);
    for (int i = 0; i < 81; i++) {
        if (puzzle[i] != '.') {
            int digit = puzzle[i] - '0';
            vec[i * 10 + digit - 1] = 1;
        }
    }
    return vec;
}

void generate_patterns(int i, int j, vector<double>& puzzle, vector<vector<double>>& patterns) {
    if (i == 9) {
        patterns.push_back(puzzle);
        return;
    }
    int next_i = i;
    int next_j = j + 1;
    if (next_j == 9) {
        next_i = i + 1;
        next_j = 0;
    }
    if (puzzle[i * 9 + j] != 0) {
        generate_patterns(next_i, next_j, puzzle, patterns);
    }
    else {
        for (int k = 1; k <= 9; k++) {
            bool is_valid = true;
            for (int l = 0; l < 9; l++) {
                if (puzzle[i * 9 + l] == k || puzzle[l * 9 + j] == k || puzzle[((i / 3) * 3 + l / 3) * 9 + (j / 3) * 3 + l % 3] == k) {
                    is_valid = false;
                    break;
                }
            }
            if (is_valid) {
                vector<double> new_puzzle = puzzle;
                new_puzzle[i * 9 + j] = k;
                generate_patterns(next_i, next_j, new_puzzle, patterns);
            }
        }
    }
}

vector<vector<double>> generate_patterns() {
    vector<vector<double>> patterns;
    vector<double> puzzle(81, 0);
    generate_patterns(0, 0, puzzle, patterns);
    return patterns;
}



  string solve_sudoku(string puzzle) {
      HopfieldNetwork net(81 * 10);

      // Generate patterns
      vector<vector<double>> patterns = generate_patterns();

      // Train network on patterns
      net.train(patterns);

      // Convert input puzzle to one-hot encoded vector
      vector<double> input = sudoku_to_vector(puzzle);

      // Recall solution from network
      vector<double> output = net.recall(input);

      // Convert output vector to string representation
      string solution = vector_to_sudoku(output);

      return solution;
  }

  int main() {
      srand(time(NULL));
      vector<string> puzzles = {
          "53..7.............4.............................................................",
          "1...67.5.8..34...........5.....2.....3...4..........267.........................",
          ".4.....6.71..9.....3.4.....7...16.3.....8..29....4.....5........................",
          "469..8..17....16...2...73..2........31......657.....2...........................",
          "..7.2.....8.63......4.....1....5..83....7.....9...15..3........................."
      };
      for (auto puzzle : puzzles) {
          cout << "Puzzle: " << puzzle << endl;
          string solution = solve_sudoku(puzzle);
          cout << "Solution: " << solution << endl;
      }
      return 0;
  }
