#include <torch/torch.h>
#include <vector>
#include <deque>
#include <random>
#include <iostream>

#include "Board.h"

// ======================== Hyperparameters ========================
const int CHANNELS = 2;        // Two layers for pieces (player and opponent)
const int HEIGHT = 6;          // Connect 4 grid height
const int WIDTH = 7;           // Connect 4 grid width
const int ACTION_SIZE = 7;     // 7 possible moves
const double GAMMA = 0.95;
const double LEARNING_RATE = 0.0005;
const int MEMORY_SIZE = 50000;
const int BATCH_SIZE = 64;
const double EPSILON_DECAY = 0.995;
const double MIN_EPSILON = 0.05;
const float GRADIENT_CLIP_VALUE = 1.f;
const float TAU = 0.005;  // Soft update parameter

// ======================== CNN-Based DQN ========================
struct DQNImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };

    DQNImpl() {
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNELS, 32, 3).stride(1).padding(1));
        register_module("conv1", conv1);

        conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1));
        register_module("conv2", conv2);

        fc1 = torch::nn::Linear(64 * HEIGHT * WIDTH, 128);
        register_module("fc1", fc1);

        fc2 = torch::nn::Linear(128, ACTION_SIZE);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1(x));
        x = torch::relu(conv2(x));
        x = x.view({ x.size(0), -1 });  // Flatten
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

// ✅ Register CNN-Based DQN as a Torch Module
TORCH_MODULE(DQN);

// ======================== Experience Replay Buffer ========================
struct ReplayBuffer {
    std::deque<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>> memory;
    std::random_device rd;
    std::mt19937 gen;

    ReplayBuffer() : gen(rd()) {}

    void push(torch::Tensor state, int action, double reward, torch::Tensor next_state, bool done) {
        if (memory.size() >= MEMORY_SIZE) {
            int removeIndex = -1;
            double lowestReward = 1.0;  // Initialize to the best possible reward

            // Find the worst experience (negative reward) to remove
            for (int i = 0; i < MEMORY_SIZE / 2; i++) {
                double reward = std::get<2>(memory[i]);  // Get the reward from memory
                if (reward < lowestReward) {
                    lowestReward = reward;
                    removeIndex = i;
                }
            }

            // Remove the worst experience found (if any)
            if (removeIndex != -1) {
                memory.erase(memory.begin() + removeIndex);
            }
            else {
                memory.pop_front();  // If no bad experiences, remove the oldest
            }
        }

        memory.emplace_back(state, action, reward, next_state, done);  // Add the individual experience directly
    }

    std::vector<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>> sample(int batch_size) {
        std::vector<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>> batch;
        std::sample(memory.begin(), memory.end(), std::back_inserter(batch), batch_size, gen);
        return batch;
    }

    bool is_ready() { return memory.size() >= BATCH_SIZE; }
};

// ======================== DQN Agent ========================
class DQNAgent {
public:
    float Loss;
    DQN policy_net;
    DQN target_net;  // Target network for stable learning
    torch::optim::Adam optimizer;
    double epsilon;

    DQNAgent()
        : policy_net(DQN()),
        target_net(DQN()),
        optimizer(policy_net->parameters(), torch::optim::AdamOptions(LEARNING_RATE)),
        epsilon(1.0) {  // Start with high exploration

        try {
            policy_net->load_model("policyRealCNN.model");
            target_net->load_model("policyRealCNN.model");
            target_net->eval();  // Set target net to evaluation mode
        }
        catch (const std::exception& e) {
            std::cout << "Model not found, initializing a new one." << std::endl;
            policy_net->save_model("policyRealCNN.model");
        }

        auto device = policy_net->parameters().front().device();
        target_net->to(device);
    }

    int select_action(torch::Tensor state, double epsilon) {
        auto device = policy_net->parameters().front().device();
        state = state.to(device);

        if (rand() < epsilon * RAND_MAX) {
            return rand() % ACTION_SIZE;
        }

        state = state.unsqueeze(0);
        auto q_values = policy_net->forward(state);
        return q_values.argmax(1).item<int>();
    }
    torch::Tensor flip_board(const torch::Tensor& board) {
        // Assuming `board` is a 2D tensor with the shape (ROWS, COLS)
        torch::Tensor flipped = board.clone();  // Clone the tensor to avoid modifying the original

        // Flip the board by inverting values: Yellow (1) becomes Red (0) and vice versa
        flipped = torch::where(flipped == 1, torch::zeros_like(flipped), flipped); // Replace Yellow (1) with 0
        flipped = torch::where(flipped == 0, torch::ones_like(flipped), flipped); // Replace Red (0) with 1

        return flipped;
    }

    void train(ReplayBuffer& buffer) {
        if (!buffer.is_ready()) return;

        auto batch = buffer.sample(BATCH_SIZE);
        auto device = policy_net->parameters().front().device();

        std::vector<torch::Tensor> states, next_states;
        std::vector<int> actions;
        std::vector<double> rewards;
        std::vector<bool> dones;
        Value winner;
        float total_loss = 0.0f;

        for (auto& [state, action, reward, next_state, done] : batch) {
            Board board;

            // Identify the winner based on the reward
            if (reward > 0) {
                winner = Value::Yellow; // AI wins
            }
            else {
                winner = Value::Red; // Opponent wins
            }

            // Flip the perspective only for the AI's loss (not for both outcomes)
            if (winner != Value::Yellow) {
                state = flip_board(state);  // Flip state for opponent's perspective
                next_state = flip_board(next_state);  // Flip next state for opponent's perspective
                reward = -reward;  // Penalize the AI for losing by reversing the reward
            }

            states.push_back(state);
            actions.push_back(action);
            rewards.push_back(reward);
            next_states.push_back(next_state);
            dones.push_back(done);

            // Optionally, store the opponent's perspective as a second sample if needed
            // This can help the AI learn both perspectives but is not mandatory in every game.
            // states.push_back(flip_board(state)); // Optional: flip board to learn from opponent’s perspective
            // actions.push_back(action); // Use the same action as the opponent's perspective
            // rewards.push_back(-reward); // Reverse the reward for opponent's perspective
            // next_states.push_back(flip_board(next_state)); // Flip next state for opponent's perspective
            // dones.push_back(done); // Same done status
        }

        // Convert vectors to tensors
        auto state_tensor = torch::stack(states).to(device);
        auto next_state_tensor = torch::stack(next_states).to(device);
        auto action_tensor = torch::tensor(actions, torch::kLong).to(device);
        auto reward_tensor = torch::tensor(rewards).to(device);
        auto done_tensor = torch::tensor(std::vector<int64_t>(dones.begin(), dones.end()), torch::kBool).to(device);

        // Compute Q-values for current states and actions
        auto q_values = policy_net->forward(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1);

        // Compute target Q-values using the target network
        auto next_q_values = std::get<0>(torch::max(target_net->forward(next_state_tensor), 1)).detach();
        auto target_q_values = reward_tensor + GAMMA * next_q_values * (~done_tensor);

        // Compute loss (Mean Squared Error loss)
        auto loss = torch::mse_loss(q_values, target_q_values);

        // Optimize model (zero gradient, backpropagate, and update)
        optimizer.zero_grad();
        loss.backward();

        // Gradient clipping to stabilize training
        for (auto& param : policy_net->parameters()) {
            if (param.grad().defined()) {
                torch::nn::utils::clip_grad_norm_(param, GRADIENT_CLIP_VALUE);
            }
        }

        optimizer.step();

        // Decay epsilon for exploration-exploitation tradeoff
        epsilon = std::max(epsilon * EPSILON_DECAY, MIN_EPSILON);

        // Update the total loss for debugging
        total_loss += loss.item<float>();
        float avg_loss = total_loss / BATCH_SIZE;
        Loss = avg_loss;  // Save the average loss for debugging purposes

        // --- Efficient learning from losses ---
        // Reward adjustment after the game ends
        if (winner == Value::Red) {
            // When the AI loses, learn from the moves that led to the loss:
            // For simplicity, just apply a moderate penalty to each action
            // in the episode leading to the loss.
            for (int i = 0; i < actions.size(); ++i) {
                rewards[i] = rewards[i] - 10.0f;  // Apply a small penalty for each action
            }
        }

        // Further adjustments can be made here depending on how you want the AI to learn
        // for example, more complex reward shaping to fine-tune the learning
    }


    void update_target() {
        for (size_t i = 0; i < policy_net->parameters().size(); ++i) {
            target_net->parameters()[i].data().mul_(1 - TAU).add_(TAU * policy_net->parameters()[i].data());
        }
    }

    void save() {
        policy_net->save_model("policyRealCNN.model");
    }

    float getLoss() { return Loss; }
};