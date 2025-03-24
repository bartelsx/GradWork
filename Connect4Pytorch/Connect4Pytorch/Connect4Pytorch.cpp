//////#include <torch/torch.h>
//////#include <iostream>
//////
//////int main() {
//////    // Print a message to verify that PyTorch is being used
//////    std::cout << "PyTorch is working!" << std::endl;
//////
//////    // Create a tensor filled with random values
//////    torch::Tensor tensor = torch::randn({ 2, 3 });  // 2x3 matrix with random values
//////    std::cout << "Tensor: " << tensor << std::endl;
//////
//////    // Check tensor's properties
//////    std::cout << "Tensor size: " << tensor.sizes() << std::endl;
//////
//////    return 0;
//////}
////// Connect4_Console.cpp : This file contains the 'main' function. Program execution begins and ends there.
//////
////
////#include <iostream>
////#include <windows.h>
////#include <conio.h>
////
////#include "Board.h"
////#include "Connect4Algorithm.h"
////
////
////#define LEVEL 4 //1 to 6
////#define MACHINE_COLOR Value::Red
////#define MACHINE_PLAYS_FIRST true
////
////#define NORMAL 7
////#define RED 12
////#define YELLOW 14
////
////void PrintBoard(const Board& board)
////{
////	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
////	system("CLS");
////	std::cout << "\n";
////	std::cout << " +---------------------------+\n";
////	std::cout << " |       Xander Bartels      |\n";
////	std::cout << " |         Connect 4         |\n";
////	std::cout << " +---------------------------+\n";
////	std::cout << "\n";
////	std::cout << " +---------------------------+\n";
////	for (int row = Board::MAX_DISCS_PER_COLUMN - 1; row >= 0; --row)
////	{
////		for (int column = 0; column < Board::COLUMNS; ++column)
////		{
////			auto value = board.GetValue(row, column);
////
////			std::cout << " | ";
////
////			SetConsoleTextAttribute(hConsole, value == Value::None ? NORMAL : value == Value::Red ? RED : YELLOW);
////			std::cout << (value == Value::None ? " " : value == Value::Red ? "R" : "Y");
////			SetConsoleTextAttribute(hConsole, NORMAL);
////		}
////		std::cout << " |\n";
////	}
////	std::cout << " +---------------------------+\n";
////	std::cout << " | 1 | 2 | 3 | 4 | 5 | 6 | 7 |\n";
////	std::cout << " +---------------------------+\n";
////}
////
////void LetMachineMakeMove(Connect4Algorithm& algorithm, Board& board)
////{
////	std::cout << "I'm thinking ...";
////
////	//let AI do its action
////	int columnToPlay = algorithm.GetNextMove(board);
////	if (columnToPlay >= 0 && columnToPlay <= Board::COLUMNS)
////	{
////		board.Drop(MACHINE_COLOR, columnToPlay);
////	}
////	else
////	{
////		std::cout << "Machine can not make a move :O ";
////	}
////
////}
////
////bool LetHumanMakeMove(Board& board)
////{
////	std::string answer;
////
////	//ask player action
////	bool validInput;
////	do {
////		validInput = true;
////		std::cout << "Enter column number you want to drop a disc in (1-7), or G to Give Up : ";
////		std::cin >> answer;
////		if (answer == "G" || answer == "g")
////		{
////			return false;
////		}
////
////		auto column = answer[0] - '1';
////		if (column >= 0 && column < Board::COLUMNS && board.GetSize(column) < Board::MAX_DISCS_PER_COLUMN)
////		{
////			board.Drop(MACHINE_COLOR == Value::Red ? Value::Yellow : Value::Red, column);
////		}
////		else
////		{
////			std::cout << "Invalid input, try again\n";
////			validInput = false;
////		}
////	} while (!validInput);
////	return true;
////}
////
////int main()
////{
////	Board board{};
////	Connect4Algorithm algorithm{ MACHINE_COLOR, LEVEL };
////
////	bool endOfGame{ false };
////	bool machineIsPlaying{ MACHINE_PLAYS_FIRST };
////
////	do
////	{
////		PrintBoard(board);
////
////		if (machineIsPlaying)
////		{
////			LetMachineMakeMove(algorithm, board);
////		}
////		else
////		{
////			if (!LetHumanMakeMove(board))
////			{
////				std::cout << "You gave up - you lose !";
////				endOfGame = true;
////			}
////		}
////		machineIsPlaying = !machineIsPlaying;
////
////		//check end of game
////		auto winner = board.HasFourInARow();
////		if (winner != Value::None)
////		{
////			endOfGame = true;
////			PrintBoard(board);
////			std::cout << "The winner is " << (winner == Value::Red ? "red" : "yellow");
////		}
////
////	} while (!endOfGame);
////}
///
#include <torch/torch.h>
#include "DQNAgent.cpp"
#include <iostream>
#include <fstream>
#include <windows.h>
#include <conio.h>
#include "Board.h"
#include "Connect4Algorithm.h"



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
	}
	else {
		std::cout << "No epsilon file found, starting from epsilon = 1" << std::endl;
		epsilon = 0.1; // If no saved epsilon, start from exploration
	}



	//dqnAI.update_target();
	for (int episode = 0; episode < 1000000; ++episode) { // Training loop
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
			}
			else 
			{
				move = minimaxAI.GetNextMove(board);
				if (!board.IsValidMove(move)) continue;
				board.Drop(MACHINE_COLOR, move);
			}

			torch::Tensor nextState = board.ToTensor();
			double reward = board.GetReward(DQN_COLOR);
			bool done = board.IsGameOver();

			// Push a single move (experience) into the game trajectory
			gameTrajectory.push_back(std::make_tuple(state, move, reward, nextState, done));
			


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

		buffer.push(gameTrajectory);

		if (buffer.is_ready()) {
			dqnAI.train(buffer);

		}

		epsilon = (epsilon * EPSILON_DECAY > MIN_EPSILON) ? (epsilon * EPSILON_DECAY) : MIN_EPSILON;

		if (episode % 200 == 0) 
		{
			torch::save(dqnAI.policy_net, "policyReal.model");
			dqnAI.update_target();  // Update target network every 500 episodes
			std::ofstream epsilon_file("epsilon.txt");
			epsilon_file << epsilon;
			epsilon_file.close();
		}
	}

	logFile.close();
	std::cout << "Training complete! Saving model..." << std::endl;
	torch::save(dqnAI.policy_net, "policyReal.model");
	return 0;
}


