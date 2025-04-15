#include <torch/torch.h>
#include <vector>
#include <deque>
#include <random>
#include <iostream>

#include "Board.h"

// ======================== Hyperparameters ========================
const int STATE_SIZE = 84;
const int ACTION_SIZE = 7;
const double GAMMA = 0.95;
const double LEARNING_RATE = 0.0005;
const int MEMORY_SIZE = 50000;
const int BATCH_SIZE = 128;
const double EPSILON_DECAY = 0.9995;
const double MIN_EPSILON = 0.05;
const float GRADIENT_CLIP_VALUE = 1.0f;
const float TAU = 0.005;

// ======================== Configure PyTorch Threading ========================


// ======================== Neural Network (DQN) ========================
struct DQNImpl : torch::nn::Module {
	// Define layers explicitly
	torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr };
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };

	DQNImpl() {
		// Convolutional layers
		conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 32, 4).stride(1).padding(1)));
		conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 4).stride(1).padding(1)));
		conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4).stride(1).padding(1)));

		// Fully connected layers
		fc1 = register_module("fc1", torch::nn::Linear(128 * 3 * 4, 128)); // Adjusted input size based on CNN output
		fc2 = register_module("fc2", torch::nn::Linear(128, ACTION_SIZE));
	}

	torch::Tensor forward(torch::Tensor x) {
		//std::cout << "Input shape: " << x.sizes() << std::endl;

		x = x.view({ x.size(0), 2, 6, 7 }); // Reshape for CNN input
	//	std::cout << "After reshape: " << x.sizes() << std::endl;

		x = torch::relu(conv1(x));
		x = torch::relu(conv2(x));
		x = torch::relu(conv3(x));
		//std::cout << "After conv layers: " << x.sizes() << std::endl;

		x = x.view({ x.size(0), -1 }); // Flatten for FC layers
	//	std::cout << "Flattened shape: " << x.sizes() << std::endl;

		x = torch::relu(fc1(x));
		x = fc2(x);

		return x;
	}


	void save_model(const std::string& file_path) {
		torch::serialize::OutputArchive output_archive;
		save(output_archive);
		output_archive.save_to(file_path);
	}

	void load_model(const std::string& file_path) {
		torch::serialize::InputArchive input_archive;
		input_archive.load_from(file_path);
		load(input_archive);
	}
};
TORCH_MODULE(DQN);

// ======================== Experience Replay Buffer ========================
struct ReplayBuffer {
	std::deque<std::vector<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>>> memory;
	std::random_device rd;
	std::mt19937 gen;

	ReplayBuffer() : gen(rd()) {}

	void push(std::vector<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>> game_trajectory) {
		if (memory.size() >= MEMORY_SIZE) memory.pop_front();
		memory.push_back(game_trajectory);
	}

	std::vector<std::vector<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>>> sample(int batch_size) {
		std::vector<std::vector<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>>> batch;
		std::sample(memory.begin(), memory.end(), std::back_inserter(batch), batch_size, gen);
		return batch;
	}

	bool is_ready() { return memory.size() >= BATCH_SIZE; }
};

// ======================== DQN Agent (Single-Threaded) ========================
class DQNAgent {
public:
	float Loss;
	DQN policy_net, target_net;
	torch::optim::Adam optimizer{ policy_net->parameters(), torch::optim::AdamOptions(LEARNING_RATE) };

	DQNAgent() {
		torch::set_num_threads(1);
		torch::set_num_interop_threads(1);
		try {
			target_net->load_model("policyReal.model");
			policy_net->load_model("policyReal.model");
			target_net->eval();
		}
		catch (const std::exception& e) {
			std::cout << "Model not found, initializing a new one." << std::endl;
			policy_net->save_model("policyReal.model");
		}
	}

	int select_action(torch::Tensor state, double epsilon) {
		state = state.view({ 1, STATE_SIZE });

		if ((rand() / double(RAND_MAX)) < epsilon) {
			return rand() % ACTION_SIZE;
		}

		auto q_values = policy_net->forward(state);
		return q_values.argmax(1).item<int>();
	}

	void train(ReplayBuffer& buffer) {
		if (!buffer.is_ready()) return;

		auto batch = buffer.sample(BATCH_SIZE);
		std::vector<torch::Tensor> loss_tensors;

		for (auto& game_trajectory : batch) {
			std::vector<torch::Tensor> states, next_states;
			std::vector<int> actions;
			std::vector<double> rewards;
			std::vector<int64_t> dones;

			for (auto& [state, action, reward, next_state, done] : game_trajectory) {
				states.push_back(state);
				actions.push_back(action);
				rewards.push_back(reward);
				next_states.push_back(next_state);
				dones.push_back(done ? 1 : 0);
			}

			Board board = ConvertTensorToBoard(states[0]);

			if (opponent_is_building_win(board, Value::Yellow)) {
				rewards[0] -= 10.0;
			}

			auto state_tensor = torch::stack(states);
			auto next_state_tensor = torch::stack(next_states);
			auto action_tensor = torch::tensor(actions, torch::kLong);
			auto reward_tensor = torch::tensor(rewards).div_(10.0);
			auto done_tensor = torch::tensor(dones, torch::kFloat);

			auto q_values = policy_net->forward(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1);
			auto next_q_values = std::get<0>(target_net->forward(next_state_tensor).max(1)).detach();
			auto target_q_values = reward_tensor + GAMMA * next_q_values * (1 - done_tensor);

			loss_tensors.push_back(torch::smooth_l1_loss(q_values, target_q_values));
		}

		auto total_loss = torch::stack(loss_tensors).mean();
		optimizer.zero_grad();
		total_loss.backward();
		torch::nn::utils::clip_grad_norm_(policy_net->parameters(), 0.05);
		optimizer.step();
		Loss = total_loss.item<float>();
	}





	float getLoss() { return Loss; }

	bool opponent_is_building_win(Board board, Value opponent_color) {
		for (int col = 0; col < Board::COLUMNS; ++col) {
			if (!board.IsValidMove(col)) continue;

			board.Drop(opponent_color, col);
			int threats = board.CalculateScore(opponent_color);
			board.RemoveDisc(col);

			if (threats >= 900) return true;
		}
		return false;
	}

	Board ConvertTensorToBoard(const torch::Tensor& state_tensor) {
		Board board;
		torch::Tensor board_tensor = state_tensor.to(torch::kCPU).to(torch::kInt);

		for (int c = 0; c < Board::COLUMNS; ++c) {
			for (int r = 0; r < Board::MAX_DISCS_PER_COLUMN; ++r) {
				int red_value = board_tensor[0][r][c].item<int>();
				int yellow_value = board_tensor[1][r][c].item<int>();

				if (red_value == 1) board.Drop(Value::Red, c);
				else if (yellow_value == 1) board.Drop(Value::Yellow, c);
			}
		}
		return board;
	}


	void update_target() {
		torch::save(policy_net, "policyReal.model");
		torch::load(target_net, "policyReal.model");
	}
};
