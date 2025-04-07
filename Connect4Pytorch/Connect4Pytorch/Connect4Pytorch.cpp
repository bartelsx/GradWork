
#include <torch/torch.h>
#include "DQNAgent.cpp"
#include <iostream>
#include <fstream>
#include <windows.h>
#include <conio.h>
#include "Board.h"+
#include "Connect4Algorithm.h"
#include <random> // ✅ Needed for random moves


#define LEVEL 0 // MiniMax depth
#define MACHINE_COLOR Value::Red // MiniMax AI
#define DQN_COLOR Value::Yellow // DQN AI

#define NORMAL 7
#define RED 12
#define YELLOW 14

void PrintBoard(const Board& board) {
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	//system("CLS");
	std::cout << "\n +---------------------------+\n";
	std::cout << " |       DQN Training       |\n";
	std::cout << " |         Connect 4        |\n";
	std::cout << " +---------------------------+\n\n";
	std::cout << " +---------------------------+\n";
	for (int row = Board::MAX_DISCS_PER_COLUMN - 1; row >= 0; --row) {
		for (int column = 0; column < Board::COLUMNS; ++column) {
			auto value = board.GetValue(row, column);
			  std::cout << " | ";
			SetConsoleTextAttribute(hConsole, value == Value::None ? NORMAL : value == Value::Red ? RED : YELLOW);
			  std::cout << (value == Value::None ? " " : value == Value::Red ? "R" : "Y");
			SetConsoleTextAttribute(hConsole, NORMAL);
		}
		 std::cout << " |\n";
	}
	std::cout << " +---------------------------+\n";
	std::cout << " | 1 | 2 | 3 | 4 | 5 | 6 | 7 |\n";
	std::cout << " +---------------------------+\n";
}

int getRandomMove(const Board& board) {
	static std::random_device rd;  // 🔥 Seed for true randomness
	static std::mt19937 gen(rd()); // ✅ Mersenne Twister PRNG
	std::vector<int> valid_moves;

	for (int col = 0; col < Board::COLUMNS; ++col) {
		if (board.IsValidMove(col)) valid_moves.push_back(col);
	}

	if (valid_moves.empty()) return -1;

	std::uniform_int_distribution<> distrib(0, valid_moves.size() - 1);  // ✅ Uniform distribution
	return valid_moves[distrib(gen)];
}
int main() {
	Board board;
	Connect4Algorithm minimaxAI(MACHINE_COLOR, LEVEL);
	DQNAgent dqnAI;
	ReplayBuffer buffer;
	bool training = true;
	double epsilon = 0.7;


	
	std::ofstream logFile("training_log.txt", std::ios::out);
	if (!logFile) {
		//    std::cerr << "Error opening log file!" << std::endl;
		return 1;
	}
	torch::load(dqnAI.policy_net, "policyReal.model");
	torch::load(dqnAI.target_net, "policyReal.model");

	std::ifstream epsilon_file("epsilon.txt");
	if (epsilon_file.is_open()) {
		epsilon_file >> epsilon;
		epsilon_file.close();
		epsilon = 0.01;
	}
	else {
		std::cout << "No epsilon file found, starting from epsilon = 1" << std::endl;
		epsilon = 1; // If no saved epsilon, start from exploration
	}



	//dqnAI.update_target();
	for (int episode = 0; episode < 1000; ++episode) { // Training loop
		//  std::cout << episode<<"\n";
		board.Reset();
		bool dqnTurn = (DQN_COLOR == Value::Red);
		bool gameOver = false;
		int moves = 0;
		//PrintBoard(board);
		std::vector<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>> gameTrajectory;

		while (!gameOver) 
		{
			int move;
			torch::Tensor state = board.ToTensor();

			


			if (dqnTurn) 
			{
				move = dqnAI.select_action(state, epsilon);
				if (!board.IsValidMove(move)) continue;
				board.Drop(DQN_COLOR, move);
				//move = getRandomMove(board); // 🔥 Random AI move
				//if (move == -1) break; // No valid moves left, game should end
				//board.Drop(DQN_COLOR, move);
			}
			else 
			{
				move = getRandomMove(board); // 🔥 Random AI move
				if (move == -1) break; // No valid moves left, game should end
				board.Drop(MACHINE_COLOR, move);
			}

			torch::Tensor nextState = board.ToTensor();
			double reward = board.GetReward(DQN_COLOR);
			bool done = board.IsGameOver();

			// Push a single move (experience) into the game trajectory
			//gameTrajectory.push_back(std::make_tuple(state, move, reward, nextState, done));
			


			moves++;
			if (done) {
				gameOver = true;
				Value winner = board.HasFourInARow();
				logFile << "Episode: " << episode
					<< ", Moves: " << moves
					<< ", Winner: " << (winner == DQN_COLOR ? "DQN" : winner == MACHINE_COLOR ? "MiniMax" : "Draw")
					<< ", Loss: " << dqnAI.getLoss()  // Add the loss here
					<< std::endl;
				//PrintBoard(board);
				break;
			}

			dqnTurn = !dqnTurn;
			//PrintBoard(board);
		}

		//buffer.push(gameTrajectory);
		//
		//if (buffer.is_ready()) {
		//	dqnAI.train(buffer);
		//
		//}
		////epsilon = max(epsilon - 0.0001, MIN_EPSILON);
		//
		//epsilon = (epsilon * EPSILON_DECAY > MIN_EPSILON) ? (epsilon * EPSILON_DECAY) : MIN_EPSILON;
		//
		//if (episode % 200 == 0) 
		//{
		//	torch::save(dqnAI.policy_net, "policyReal.model");
		//	dqnAI.update_target();  // Update target network every 500 episodes
		//	std::ofstream epsilon_file("epsilon.txt");
		//	epsilon_file << epsilon;
		//	epsilon_file.close();
		//}
	}

	logFile.close();
	std::cout << "Training complete! Saving model..." << std::endl;
	//torch::save(dqnAI.policy_net, "policyReal.model");
	return 0;
}


