#include <torch/torch.h>
#include <vector>
#include <deque>
#include <random>
#include <iostream>

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
const float GRADIENT_CLIP_VALUE = 1.0f;
const float TAU = 0.005;  // Soft update parameter

// ======================== CNN-Based DQN ========================
struct DQNImpl : torch::nn::Module {
    // Convolutional layers
    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };

    // Fully connected layers
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
    DQN policy_net;
    DQN target_net;  // Target network for stable learning
    torch::optim::Adam optimizer;

    DQNAgent()
        : policy_net(DQN()),
        target_net(DQN()),
        optimizer(policy_net->parameters(), torch::optim::AdamOptions(LEARNING_RATE)) {

        try {
            target_net->load_model("policyRealCNN.model");
            policy_net->load_model("policyRealCNN.model");
            target_net->eval();  // Set target net to evaluation mode
        }
        catch (const std::exception& e) {
            std::cout << "Model not found, initializing a new one." << std::endl;
            policy_net->save_model("policyRealCNN.model");
        }

        auto device = policy_net->parameters().front().device();  // Get device
        target_net->to(device);
    }

    int select_action(torch::Tensor state, double epsilon) {
        auto device = policy_net->parameters().front().device();
        state = state.to(device);

        if (rand() < epsilon * RAND_MAX) {
            return rand() % ACTION_SIZE;  // Random action (exploration)
        }

        state = state.unsqueeze(0);  // Add batch dimension
        auto q_values = policy_net->forward(state);
        return q_values.argmax(1).item<int>();  // Best action
    }

    void train(ReplayBuffer& buffer) {
        if (!buffer.is_ready()) return;

        auto batch = buffer.sample(BATCH_SIZE);
        auto device = policy_net->parameters().front().device();

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

            auto q_values = policy_net->forward(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1);
            auto next_q_output = target_net->forward(next_state_tensor);
            auto next_q_values = std::get<0>(torch::max(next_q_output, 1)).detach();

            auto target_q_values = reward_tensor + GAMMA * next_q_values * (~done_tensor);

            auto loss = torch::mse_loss(q_values, target_q_values);
            Loss = loss.item<float>();

            optimizer.zero_grad();
            loss.backward();

            for (auto& param : policy_net->parameters()) {
                if (param.grad().defined()) {
                    torch::nn::utils::clip_grad_norm_(param, GRADIENT_CLIP_VALUE);
                }
            }

            optimizer.step();
        }
    }

    void update_target() {
        target_net->load_model("policyRealCNN.model");
    }

    float getLoss() { return Loss; }
};
