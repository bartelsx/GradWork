////#include <torch/torch.h>
////#include <vector>
////#include <deque>
////#include <random>
////
////// ======================== Hyperparameters ========================
////const int STATE_SIZE = 84; // 2* 6 rows x 7 columns for Connect 4
////const int ACTION_SIZE = 7; // 7 possible moves
////const double GAMMA = 0.99;
////const double LEARNING_RATE = 0.0005;//0.001 old
////const int MEMORY_SIZE = 50000;
////const int BATCH_SIZE = 64;
////const double EPSILON_DECAY = 0.995;//0.995
////const double MIN_EPSILON = 0.05;//0.01
////
////// ======================== Neural Network (DQN) ========================
////struct DQNImpl : torch::nn::Module {
////    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
////
////    DQNImpl() {
////        fc1 = torch::nn::Linear(STATE_SIZE, 128);
////        register_module("fc1", fc1);
////
////        fc2 = torch::nn::Linear(128, 128);
////        register_module("fc2", fc2);
////
////        fc3 = torch::nn::Linear(128, ACTION_SIZE);
////        register_module("fc3", fc3);
////    }
////
////
////    torch::Tensor forward(torch::Tensor x) {
////      //  std::cout << "Before reshaping, tensor shape: " << x.sizes() << std::endl;
////
////        // Flatten the tensor from [1, 2, 6, 7] to [1, 84]
////        x = x.view({ x.size(0), -1 });  // This will flatten all dimensions except the batch size
////      //  std::cout << "After reshaping, tensor shape: " << x.sizes() << std::endl;  // Should print [1, 84]
////
////        // Apply the fully connected layers
////        x = torch::relu(fc1(x));  // Apply first layer with ReLU activation
////        //std::cout << "After fc1, tensor shape: " << x.sizes() << std::endl;  // Should print [1, 128]
////
////        x = torch::relu(fc2(x));  // Apply second layer with ReLU activation
////        x = fc3(x);  // Output Q-values from the final layer
////        return x;
////    }
////
////
////
////    // ✅ Save model function
////    void save_model(const std::string& file_path) {
////        torch::serialize::OutputArchive output_archive;
////        save(output_archive);
////        output_archive.save_to(file_path);
////    }
////
////    // ✅ Load model function
////    void load_model(const std::string& file_path) {
////        torch::serialize::InputArchive input_archive;
////        input_archive.load_from(file_path);
////        load(input_archive);
////    }
////};
////
////// ✅ Register DQN as a proper PyTorch module
////TORCH_MODULE(DQN);
////
////// ======================== Experience Replay Buffer ========================
////struct ReplayBuffer {
////    std::deque<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>> memory;
////    std::random_device rd;
////    std::mt19937 gen;
////
////    ReplayBuffer() : gen(rd()) {}
////
////    void push(torch::Tensor state, int action, double reward, torch::Tensor next_state, bool done) {
////        if (memory.size() >= MEMORY_SIZE) memory.pop_front();
////        memory.push_back({ state, action, reward, next_state, done });
////    }
////
////    std::vector<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>> sample(int batch_size) {
////        std::vector<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>> batch;
////        std::sample(memory.begin(), memory.end(), std::back_inserter(batch), batch_size, gen);
////        return batch;
////    }
////
////    bool is_ready() { return memory.size() >= BATCH_SIZE; }
////};
////
////// ======================== DQN Agent ========================
////class DQNAgent {
////public:
////    DQNAgent()
////        : policy_net(DQN()),
////        target_net(DQN()),
////        optimizer(policy_net->parameters(), torch::optim::AdamOptions(LEARNING_RATE)) {
////        try {
////            target_net->load_model("policy.model");
////            target_net->eval(); // Set target net to evaluation mode
////        }
////        catch (const std::exception& e) {
////         //   std::cout << "Model not found, initializing a new one." << std::endl;
////            // Optionally, save the model if not found (only for the first run)
////            policy_net->save_model("policy.model");
////        }
////        auto device = policy_net->parameters().front().device();  // Get device from policy net
////        target_net->to(device); 
////    }
////
////    //int select_action(torch::Tensor state, double epsilon) {
////    //    if (((double)rand() / RAND_MAX) < epsilon) {
////    //        return rand() % ACTION_SIZE; // Random action (exploration)
////    //    }
////    //    auto q_values = policy_net->forward(state.unsqueeze(0));
////    //    return q_values.argmax(1).item<int>(); // Choose best action
////    //}
////
////    int select_action(torch::Tensor state, double epsilon) {
////        // Ensure the state tensor is on the same device as the policy network
////        auto device = policy_net->parameters().front().device(); // Get the device from the model's parameters
////        state = state.to(device); // Move the state tensor to the same device as the model
////
////        // Exploration: choose a random action with probability epsilon
////        if (((double)rand() / RAND_MAX) < epsilon) {
////            return rand() % ACTION_SIZE; // Random action (exploration)
////        }
////
////        // Flatten the state tensor to match the expected input shape [1, STATE_SIZE]
////        state = state.view({ 1, STATE_SIZE }); // Flatten and add batch dimension: shape [1, STATE_SIZE]
////
////        // Debug: Print tensor shape
////       // std::cout << "state_tensor shape after view: " << state.sizes() << std::endl;
////
////        // Check if there are any NaN values in the state tensor
////        if (state.isnan().any().item<bool>()) {
////          //  std::cerr << "Error: state_tensor contains NaN values!" << std::endl;
////            return -1; // Handle error or return an invalid action
////        }
////
////        // Move model to the correct device
////        policy_net->to(device);
////
////        // Forward pass through the policy network
////        auto q_values = policy_net->forward(state);
////
////        // Ensure q_values is not empty and has the correct shape
////        if (q_values.numel() == 0) {
////        //    std::cerr << "Error: q_values is empty!" << std::endl;
////            return -1; // Handle error or return an invalid action
////        }
////
////        // Choose the best action (greedy policy)
////        return q_values.argmax(1).item<int>(); // Choose best action
////    }
////
////
////
////
//////    void train(ReplayBuffer& buffer) {
//////        if (!buffer.is_ready()) return;
//////        auto batch = buffer.sample(BATCH_SIZE);
//////
//////        std::vector<torch::Tensor> states, next_states;
//////        std::vector<int> actions;
//////        std::vector<double> rewards;
//////        std::vector<bool> dones;
//////
//////        for (auto& [s, a, r, ns, d] : batch) {
//////            states.push_back(s);
//////            actions.push_back(a);
//////            rewards.push_back(r);
//////            next_states.push_back(ns);
//////            dones.push_back(d);
//////        }
//////
//////        auto state_tensor = torch::stack(states);
//////        auto next_state_tensor = torch::stack(next_states);
//////        auto action_tensor = torch::tensor(actions, torch::kLong);
//////        auto reward_tensor = torch::tensor(rewards);
//////        auto done_tensor = torch::tensor(std::vector<int64_t>(dones.begin(), dones.end()), torch::kBool);
//////
//////
//////        // Compute Q values
//////        auto q_values = policy_net->forward(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1);
//////        auto next_q_output = target_net->forward(next_state_tensor);
//////#undef max  // Undefine the macro definition of max
//////        auto max_result = torch::max(next_q_output, 1); // This returns a tuple (values, indices)
//////        auto next_q_values = std::get<0>(max_result).detach(); // Extract the maximum values
//////
//////
//////        auto target_q_values = reward_tensor + GAMMA * next_q_values * (~done_tensor);
//////
//////        // Compute loss and optimize
//////        auto loss = torch::mse_loss(q_values, target_q_values);
//////        optimizer.zero_grad();
//////        loss.backward();
//////        optimizer.step();
//////    }
////
////
////    void train(ReplayBuffer& buffer) {
////        if (!buffer.is_ready()) return;
////
////        auto batch = buffer.sample(BATCH_SIZE);
////        auto device = policy_net->parameters().front().device();  // Get device from policy_net
////
////        for (auto& [state, action, reward, next_state, done] : batch) {
////            // Ensure tensors are on the correct device
////            auto state_tensor = state.unsqueeze(0).to(device);        // Shape: [1, 2, 6, 7]
////            auto next_state_tensor = next_state.unsqueeze(0).to(device); // Shape: [1, 2, 6, 7]
////
////            auto action_tensor = torch::tensor({ action }, torch::kLong).to(device);
////            auto reward_tensor = torch::tensor({ reward }).to(device);
////            auto done_tensor = torch::tensor({ done }, torch::kBool).to(device);
////
////          //  std::cout << "policy_net parameters: " << policy_net->parameters().size() << std::endl;
////
////            // Compute Q value for current state using policy_net
////            auto q_values = policy_net->forward(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1);
////           // std::cout << "q_values parameters: " << policy_net->parameters().size() << std::endl;
////
////            // Compute Q values for next state using target_net
////            //std::cout << "Next state tensor shape: " << next_state_tensor.sizes() << std::endl;
////            //std::cout << "Target network parameters: " << target_net->parameters().size() << std::endl;
////
////            // Ensure target_net is on the same device as policy_net
////            auto next_q_output = target_net->forward(next_state_tensor.to(device)); // Move next_state_tensor to the correct device if necessary
////#undef max
////            auto next_q_values = std::get<0>(torch::max(next_q_output, 1)).detach(); // Detach to avoid backpropagation through target_net
////
////            // Compute target Q value
////            auto target_q_values = reward_tensor + GAMMA * next_q_values * (~done_tensor);
////
////            // Compute loss and optimize
////            auto loss = torch::mse_loss(q_values, target_q_values);
////            optimizer.zero_grad();
////            loss.backward();
////            optimizer.step();
////        }
////    }
////
////
////    void update_target() {
////        // ✅ Save policy model
////        policy_net->save_model("policy.model");
////
////        // ✅ Load into target model
////        target_net->load_model("policy.model");
////    }
////    DQN policy_net;
////
////private:
////    DQN target_net;  // Properly instantiate the model
////
////    torch::optim::Adam optimizer;
////};
////
//
//
//#include <torch/torch.h>
//#include <vector>
//#include <deque>
//#include <random>
//#include <iostream>
//
//// ======================== Hyperparameters ========================
//const int STATE_SIZE = 84; // 2 * 6 rows x 7 columns for Connect 4
//const int ACTION_SIZE = 7; // 7 possible moves
//const double GAMMA = 0.99;
//const double LEARNING_RATE = 0.0005; // 0.001 old
//const int MEMORY_SIZE = 50000;
//const int BATCH_SIZE = 64;
//const double EPSILON_DECAY = 0.995; // 0.995
//const double MIN_EPSILON = 0.05; // 0.01
//
//// ======================== Neural Network (DQN) ========================
//struct DQNImpl : torch::nn::Module {
//    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
//
//    DQNImpl() {
//        fc1 = torch::nn::Linear(STATE_SIZE, 128);
//        register_module("fc1", fc1);
//
//        fc2 = torch::nn::Linear(128, 128);
//        register_module("fc2", fc2);
//
//        fc3 = torch::nn::Linear(128, ACTION_SIZE);
//        register_module("fc3", fc3);
//    }
//
//    torch::Tensor forward(torch::Tensor x) {
//        x = x.view({ x.size(0), -1 });  // Flatten the tensor from [1, 2, 6, 7] to [1, 84]
//        x = torch::relu(fc1(x));  // Apply first layer with ReLU activation
//        x = torch::relu(fc2(x));  // Apply second layer with ReLU activation
//        x = fc3(x);  // Output Q-values from the final layer
//        return x;
//    }
//
//    void save_model(const std::string& file_path) {
//        torch::serialize::OutputArchive output_archive;
//        save(output_archive);
//        output_archive.save_to(file_path);
//    }
//
//    void load_model(const std::string& file_path) {
//        torch::serialize::InputArchive input_archive;
//        input_archive.load_from(file_path);
//        load(input_archive);
//    }
//};
//
//// ✅ Register DQN as a proper PyTorch module
//TORCH_MODULE(DQN);
//
//// ======================== Experience Replay Buffer ========================
//struct ReplayBuffer {
//    std::deque<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>> memory;
//    std::random_device rd;
//    std::mt19937 gen;
//
//    ReplayBuffer() : gen(rd()) {}
//
//    void push(torch::Tensor state, int action, double reward, torch::Tensor next_state, bool done) {
//        if (memory.size() >= MEMORY_SIZE) memory.pop_front();
//        memory.push_back({ state, action, reward, next_state, done });
//    }
//
//    std::vector<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>> sample(int batch_size) {
//        std::vector<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>> batch;
//        std::sample(memory.begin(), memory.end(), std::back_inserter(batch), batch_size, gen);
//        return batch;
//    }
//
//    bool is_ready() { return memory.size() >= BATCH_SIZE; }
//};
//
//// ======================== DQN Agent ========================
//class DQNAgent {
//public:
//    DQNAgent()
//        : policy_net(DQN()),
//        target_net(DQN()),
//        optimizer(policy_net->parameters(), torch::optim::AdamOptions(LEARNING_RATE)) {
//        try {
//            target_net->load_model("policyReal.model");
//            target_net->eval(); // Set target net to evaluation mode
//        }
//        catch (const std::exception& e) {
//            std::cout << "Model not found, initializing a new one." << std::endl;
//            policy_net->save_model("policyReal.model");
//        }
//        auto device = policy_net->parameters().front().device();  // Get device from policy net
//        target_net->to(device);
//    }
//
//    int select_action(torch::Tensor state, double epsilon) {
//        auto device = policy_net->parameters().front().device(); // Get the device from the model's parameters
//        state = state.to(device); // Move the state tensor to the same device as the model
//
//        if (((double)rand() / RAND_MAX) < epsilon) 
//        {
//           // std::cout << "random-values: " << std::endl;
//        	return rand() % ACTION_SIZE; // Random action (exploration)
//           
//        }
//
//        state = state.view({ 1, STATE_SIZE }); // Flatten and add batch dimension: shape [1, STATE_SIZE]
//
//        // Forward pass through the policy network
//        auto q_values = policy_net->forward(state);
//
//        // Log the Q-values
//       // std::cout << "Q-values: " << q_values << std::endl;
//
//        return q_values.argmax(1).item<int>(); // Choose the best action (greedy policy)
//    }
//
//    void train(ReplayBuffer& buffer) {
//        if (!buffer.is_ready()) return;
//
//        auto batch = buffer.sample(BATCH_SIZE);
//        auto device = policy_net->parameters().front().device();  // Get device from policy_net
//
//        for (auto& [state, action, reward, next_state, done] : batch) {
//            auto state_tensor = state.unsqueeze(0).to(device);
//            auto next_state_tensor = next_state.unsqueeze(0).to(device);
//            auto action_tensor = torch::tensor({ action }, torch::kLong).to(device);
//            auto reward_tensor = torch::tensor({ reward }).to(device);
//            auto done_tensor = torch::tensor({ done }, torch::kBool).to(device);
//
//            // Compute Q value for current state using policy_net
//            auto q_values = policy_net->forward(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1);
//
//            // Compute Q values for next state using target_net
//            auto next_q_output = target_net->forward(next_state_tensor);
//#undef max
//            auto next_q_values = std::get<0>(torch::max(next_q_output, 1)).detach();
//
//            // Compute target Q value
//            auto target_q_values = reward_tensor + GAMMA * next_q_values * (~done_tensor);
//
//            // Compute loss and optimize
//            auto loss = torch::mse_loss(q_values, target_q_values);
//
//            // Log the loss
//           // std::cout << "Loss: " << loss.item<double>() << std::endl;
//
//            optimizer.zero_grad();
//            loss.backward();
//            optimizer.step();
//        }
//    }
//
//    void update_target() {
//        // Save policy model
//        policy_net->save_model("policyReal.model");
//
//        // Load into target model
//        target_net->load_model("policyReal.model");
//    }
//
//    DQN policy_net;
//
//private:
//    DQN target_net;  // Properly instantiate the model
//    torch::optim::Adam optimizer;
//};

#include <torch/torch.h>
#include <vector>
#include <deque>
#include <random>
#include <iostream>

// ======================== Hyperparameters ========================
const int STATE_SIZE = 84; // 2 * 6 rows x 7 columns for Connect 4
const int ACTION_SIZE = 7; // 7 possible moves
const double GAMMA = 0.95;
const double LEARNING_RATE = 0.0005; // 0.001 old
const int MEMORY_SIZE = 50000;
const int BATCH_SIZE = 64;
const double EPSILON_DECAY = 0.995; // 0.995
const double MIN_EPSILON = 0.05; // 0.01
const float GRADIENT_CLIP_VALUE = 1.0f; // Gradient clipping threshold
const float TAU = 0.005; // Soft update parameter



// ======================== Neural Network (DQN) ========================
struct DQNImpl : torch::nn::Module {
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };

	DQNImpl() {
		fc1 = torch::nn::Linear(STATE_SIZE, 128);
		register_module("fc1", fc1);

		fc2 = torch::nn::Linear(128, 128);
		register_module("fc2", fc2);

		fc3 = torch::nn::Linear(128, ACTION_SIZE);
		register_module("fc3", fc3);
	}

	torch::Tensor forward(torch::Tensor x) {
		x = x.view({ x.size(0), -1 });  // Flatten the tensor from [1, 2, 6, 7] to [1, 84]
		x = torch::relu(fc1(x));  // Apply first layer with ReLU activation
		x = torch::relu(fc2(x));  // Apply second layer with ReLU activation
		x = fc3(x);  // Output Q-values from the final layer
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

// ✅ Register DQN as a proper PyTorch module
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

// ======================== DQN Agent ========================
class DQNAgent {
public:
	float Loss;
	DQNAgent()
		: policy_net(DQN()),
		target_net(DQN()),
		optimizer(policy_net->parameters(), torch::optim::AdamOptions(LEARNING_RATE)) {
		try {
			target_net->load_model("policyReal.model");
			policy_net->load_model("policyReal.model");
			target_net->eval(); // Set target net to evaluation mode
		}
		catch (const std::exception& e) {
			std::cout << "Model not found, initializing a new one." << std::endl;
			policy_net->save_model("policyReal.model");
		}
		auto device = policy_net->parameters().front().device();  // Get device from policy net
		target_net->to(device);
	}

	int select_action(torch::Tensor state, double epsilon) {
		auto device = policy_net->parameters().front().device(); // Get the device from the model's parameters
		state = state.to(device); // Move state to correct device

		// Exploration: choose a random action with probability epsilon
		if (rand() < epsilon * RAND_MAX) {
			return rand() % ACTION_SIZE; // Random action (exploration)
		}

		state = state.view({ 1, STATE_SIZE }); // Flatten state for input to model
		auto q_values = policy_net->forward(state);
		return q_values.argmax(1).item<int>(); // Choose best action
	}
#undef max
#undef min
	void train(ReplayBuffer& buffer) {
		if (!buffer.is_ready()) return;

		auto batch = buffer.sample(BATCH_SIZE);
		auto device = policy_net->parameters().front().device();  // Get device from policy_net

		for (auto& game_trajectory : batch) {
			std::vector<torch::Tensor> states, next_states;
			std::vector<int> actions;
			std::vector<double> rewards;
			std::vector<bool> dones;

			for (auto& [state, action, reward, next_state, done] : game_trajectory) {
				states.push_back(state);
				actions.push_back(action);
				rewards.push_back(reward);
				next_states.push_back(next_state);
				dones.push_back(done);

			}

			auto state_tensor = torch::stack(states);
			auto next_state_tensor = torch::stack(next_states);
			auto action_tensor = torch::tensor(actions, torch::kLong);
			auto reward_tensor = torch::tensor(rewards);
			auto done_tensor = torch::tensor(std::vector<int64_t>(dones.begin(), dones.end()), torch::kBool);



			// Compute Q values for the current states using policy_net
			auto q_values = policy_net->forward(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1);

			// Compute Q values for the next states using target_net
			auto next_q_output = target_net->forward(next_state_tensor);
			auto next_q_values = std::get<0>(torch::max(next_q_output, 1)).detach();

			// Compute target Q values
			auto target_q_values = reward_tensor + GAMMA * next_q_values * (~done_tensor);

			// Compute loss and optimize
			auto loss = torch::mse_loss(q_values, target_q_values);
			Loss = loss.item<float>();
			/*   std::cout << "state_tensor shape: " << state_tensor.sizes() << std::endl;
			   std::cout << "action_tensor shape: " << action_tensor.sizes() << std::endl;
			   std::cout << "action_tensor values: " << action_tensor << std::endl;
			   auto q_output = policy_net->forward(state_tensor);
			   std::cout << "q_output shape: " << q_output.sizes() << std::endl;
			   std::cout << "q_output: " << q_output << std::endl;
			   auto q_valuesdebug = q_output.gather(1, action_tensor.unsqueeze(1)).squeeze(1);
			   std::cout << "q_values: " << q_valuesdebug << std::endl;*/

			   //std::cout << "[Before Update] Q-values - Mean: " << q_values.mean().item<float>()
			   //   << " | Min: " << q_values.min().item<float>()
			   //   << " | Max: " << q_values.max().item<float>() << std::endl;

			 //  std::cout << "[Before Update] Target Q-values - Mean: " << target_q_values.mean().item<float>()
			 //     << " | Min: " << target_q_values.min().item<float>()
			 //     << " | Max: " << target_q_values.max().item<float>() << std::endl;
			 //  
			 // std::cout << "Rewards: " << reward_tensor.mean().item<float>()
			 //    << " | Min: " << reward_tensor.min().item<float>()
			 //    << " | Max: " << reward_tensor.max().item<float>() << std::endl;

			//std::cout << "Loss: " << loss.item<float>() << std::endl;

			optimizer.zero_grad();
			loss.backward();

			// Apply gradient clipping
			for (auto& param : policy_net->parameters()) {
				if (param.grad().defined()) {
					torch::nn::utils::clip_grad_norm_(param, GRADIENT_CLIP_VALUE);
				}
			}

			optimizer.step();

			// Recompute Q-values after update
			auto new_q_values = policy_net->forward(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1);

			//std::cout << "[After Update] Q-values - Mean: " << new_q_values.mean().item<float>()
			//    << " | Min: " << new_q_values.min().item<float>()
			//    << " | Max: " << new_q_values.max().item<float>() << std::endl;
		}
	}
	float getLoss()
	{
		return Loss;
	}
	void update_target() {
		// ✅ Save policy model
	//	policy_net->save_model("policyReal.model");
		target_net->load_model("policyReal.model");
		// ✅ Soft update target model using Polyak averaging
		//for (size_t i = 0; i < target_net->parameters().size(); i++) {
		//	target_net->parameters()[i] = TAU * policy_net->parameters()[i] + (1 - TAU) * target_net->parameters()[i];
		//}
	}

	DQN policy_net;
	DQN target_net;  // Target network for stable learning
private:

	torch::optim::Adam optimizer;
};
