#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

class HopfieldNetwork {
private:
    int num_neurons;
    vector<vector<double>> weights;
    vector<double> threshold;

public:
    HopfieldNetwork(int num_neurons) {
        this->num_neurons = num_neurons;
        this->weights = vector<vector<double>>(num_neurons, vector<double>(num_neurons, 0));
        this->threshold = vector<double>(num_neurons, 0);
    }

    void train(vector<vector<double>> patterns) {
        for (int i = 0; i < num_neurons; i++) {
            for (int j = 0; j < num_neurons; j++) {
                if (i == j) {
                    weights[i][j] = 0;
                }
                else {
                    for (int k = 0; k < patterns.size(); k++) {
                        weights[i][j] += patterns[k][i] * patterns[k][j];
                    }
                    weights[i][j] /= num_neurons;
                }
            }
            threshold[i] = 0;
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
                new_output[i] = activation >= 0 ? 1 : -1;
            }
            if (prev_output == new_output) {
                break;
            }
            prev_output = new_output;
        }
        return new_output;
    }

};

vector<double> sudoku_to_vector(string puzzle) {
    vector<double> vec(81, 0);
    for (int i = 0; i < puzzle.length(); i++) {
        if (puzzle[i] != '.') {
            vec[i] = (double)(puzzle[i] - '0');
        }
    }
    return vec;

}
vector<vector<double>> generate_patterns() {
    // Generate all possible Sudoku grids that satisfy the constraints
    vector<vector<double>> patterns;
    vector<double> pattern(81, 0);

    // Generate all rows
    for (int i = 0; i < 9; i++) {
        for (int j1 = 1; j1 <= 9; j1++) {
            for (int j2 = 1; j2 <= 9; j2++) {
                if (j1 != j2) {
                    pattern[i * 9 + j1 - 1] = 1;
                    pattern[i * 9 + j2 - 1] = 1;
                    patterns.push_back(pattern);
                    pattern[i * 9 + j1 - 1] = 0;
                    pattern[i * 9 + j2 - 1] = 0;
                }
            }
        }
        for (int j = 1; j <= 9; j++) {
            pattern[i * 9 + j - 1] = 1;
            patterns.push_back(pattern);
            cout << "Generated pattern: ";
            for (int k = 0; k < pattern.size(); k++) {
                cout << pattern[k];
            }
            cout << endl;
            pattern = vector<double>(81, 0);
        }

        // Generate all columns
        for (int j = 0; j < 9; j++) {
            for (int i1 = 1; i1 <= 9; i1++) {
                for (int i2 = 1; i2 <= 9; i2++) {
                    if (i1 != i2) {
                        pattern[(i1 - 1) * 9 + j] = 1;
                        pattern[(i2 - 1) * 9 + j] = 1;
                        patterns.push_back(pattern);
                        pattern[(i1 - 1) * 9 + j] = 0;
                        pattern[(i2 - 1) * 9 + j] = 0;
                    }
                }
            }
            for (int i = 1; i <= 9; i++) {
                pattern[(i - 1) * 9 + j] = 1;
            }
            patterns.push_back(pattern);
            cout << "Generated pattern: ";
            for (int k = 0; k < pattern.size(); k++) {
                cout << pattern[k];
            }
            cout << endl;
            pattern = vector<double>(81, 0);
        }

        // Generate all 3x3 subgrids
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                for (int i1 = r * 3 + 1; i1 <= r * 3 + 3; i1++) {
                    for (int j1 = c * 3 + 1; j1 <= c * 3 + 3; j1++) {
                        for (int i2 = r * 3 + 1; i2 <= r * 3 + 3; i2++) {
                            for (int j2 = c * 3 + 1; j2 <= c * 3 + 3; j2++) {
                                if (i1 != i2 || j1 != j2) {

                                    pattern[(i1 - 1) * 9 + j1 - 1] = 1;
                                    pattern[(i2 - 1) * 9 + j2 - 1] = 1;
                                    patterns.push_back(pattern);
                                    pattern[(i1 - 1) * 9 + j1 - 1] = 0;
                                    pattern[(i2 - 1) * 9 + j2 - 1] = 0;
                                }
                            }
                        }
                    }
                }
                for (int i = r * 3 + 1; i <= r * 3 + 3; i++) {
                    for (int j = c * 3 + 1; j <= c * 3 + 3; j++) {
                        pattern[(i - 1) * 9 + j - 1] = 1;
                    }
                }
                patterns.push_back(pattern);
                cout << "Generated pattern: ";
                for (int k = 0; k < pattern.size(); k++) {
                    cout << pattern[k];
                }
                cout << endl;
                pattern = vector<double>(81, 0);
            }
        }

        return patterns;
    }
}

string vector_to_sudoku(vector<double> vec) {
    string puzzle = "";
    for (int i = 0; i < vec.size(); i++) {
        if (vec[i] == 0) {
            puzzle += ".";
        }
        else {
            puzzle += to_string((int)vec[i]);
        }
        if ((i + 1) % 9 == 0) {
            puzzle += "\n";
        }
    }

    // Fill in the empty cells with the corresponding digits
    for (int i = 0; i < 81; i++) {
        if (puzzle[i] == '.') {
            for (int j = 1; j <= 9; j++) {
                vector<double> test = vec;
                test[i] = j;
                HopfieldNetwork network(81);
                network.train(generate_patterns());
                vector<double> output = network.recall(test);
                bool valid = true;
                for (int k = 0; k < 81; k++) {
                    if (puzzle[k] != '.') {
                        if (output[k] != puzzle[k] - '0') {
                            valid = false;
                            break;
                        }
                    }
                }
                if (valid) {
                    puzzle[i] = '0' + j;
                    break;
                }
            }
        }
    }

    return puzzle;
}


int main() {
    // Generate a set of Sudoku puzzles with their solutions
    vector<pair<string, string>> puzzles = {
        {"......6..7...4.8..8.7.9....4....2...7.4.......8.6...7....3.1....8.9..1.9...6....", "145239687792186543683574912926317458518462379374895126459723861237648195861951234"},
        {"8.3.7..45.95..3.7...1.86..8..67.....7..2..9...9.3.....82..16.7...2.18..12..9.7.6", "183769254596342817724518639835276491471935268269481375947853126612497583358126947"},
        {"1..9.8...7...6...2..6.7.3.8..3..7...6..1..5...1...3.9..8.4.6..4...3...7...6..8.2", "123958476745163982689724315817632549964517823532489761276345198458291637391876254"},
        {"...53.....8......2..7..1.5..4....53..9..7...6..24....8..1..6..5......4.....7...", "497538216681279435235461879146927583923856741578314692752193468364782159819645327"},
        {".9.6..7.3.3..1.4.6..8..2.7.6.2...4...7.9.8.2...9.6.1.4.7..6..8.4.1..2.4.6..7.9.", "598621734347985216126473985763152498415897623289364571931248567872536149654719832"},
        {".1.6....98...43....7...8...8.2.7.6.1.......5.7.8.9.4.3...1...7....81...72....3.8.", "314629758985417632276538491852796143143852976697341825439285167761943285528176349"},
        {"2...9.3...3.24.7......8.5.5.....6.7...6...5.1.....8.2.9.3......9.87.5...6.8...2", "276598341531247896489613257952186734347925618168734529713452968825369174694871523"},
        {".....98.1....2.7....8.9..23.9..1.8...8...7...7.6..4.51..4.6....4.7.5.....9....", "256739841918624735734815926623947158185263497479581362391476582842159673567328419"},
        {"..6....91.7....8....1..2.6.3..4.2..7.8.6.4..9.3..6.4.1..8....5....3.9..2....", "426537891197284356583169427319756284854912673762843519235691748671428935948375162"}
    };

    for (const auto& puzzle : puzzles) {
        HopfieldNetwork sudoku(81);
        vector<vector<double>> patterns = generate_patterns();
        sudoku.train(patterns);
        vector<double> input = sudoku_to_vector(puzzle.first);
        vector<double> solution = sudoku.recall(input);
        string solution_str = vector_to_sudoku(solution);
        cout << "Input Puzzle: " << puzzle.first << endl;
        cout << "Solution: " << endl;
        for (int i = 0; i < 81; i++) {
            cout << solution_str[i];
            if ((i + 1) % 9 == 0) {
                cout << endl;
            }
        }
        cout << endl;
    }

    return 0;
}


