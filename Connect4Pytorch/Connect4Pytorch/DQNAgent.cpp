#include <torch/torch.h>
#include <vector>
#include <deque>
#include <random>

// ======================== Hyperparameters ========================
const int STATE_SIZE = 42; // 6 rows x 7 columns for Connect 4
const int ACTION_SIZE = 7; // 7 possible moves
const double GAMMA = 0.99;
const double LEARNING_RATE = 0.001;
const int MEMORY_SIZE = 10000;
const int BATCH_SIZE = 64;
const double EPSILON_DECAY = 0.995;
const double MIN_EPSILON = 0.01;

// ======================== Neural Network (DQN) ========================
struct DQNImpl : torch::nn::Module {
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };

    DQNImpl() {
        fc1 = register_module("fc1", torch::nn::Linear(STATE_SIZE, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 128));
        fc3 = register_module("fc3", torch::nn::Linear(128, ACTION_SIZE));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x); // Output Q-values for each action
    }

    // ✅ Save model function
    void save_model(const std::string& file_path) {
        torch::serialize::OutputArchive output_archive;
        save(output_archive);
        output_archive.save_to(file_path);
    }

    // ✅ Load model function
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
    std::deque<std::tuple<torch::Tensor, int, double, torch::Tensor, bool>> memory;
    std::random_device rd;
    std::mt19937 gen;

    ReplayBuffer() : gen(rd()) {}

    void push(torch::Tensor state, int action, double reward, torch::Tensor next_state, bool done) {
        if (memory.size() >= MEMORY_SIZE) memory.pop_front();
        memory.push_back({ state, action, reward, next_state, done });
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
    DQNAgent()
        : policy_net(DQN()),
        target_net(DQN()),
        optimizer(policy_net->parameters(), torch::optim::AdamOptions(LEARNING_RATE)) {

        // ✅ Load model if exists
        target_net->load_model("target_model.pth");
        target_net->eval(); // Set target net to evaluation mode
    }

    int select_action(torch::Tensor state, double epsilon) {
        if (((double)rand() / RAND_MAX) < epsilon) {
            return rand() % ACTION_SIZE; // Random action (exploration)
        }
        auto q_values = policy_net->forward(state.unsqueeze(0));
        return q_values.argmax(1).item<int>(); // Choose best action
    }

    void train(ReplayBuffer& buffer) {
        if (!buffer.is_ready()) return;
        auto batch = buffer.sample(BATCH_SIZE);

        std::vector<torch::Tensor> states, next_states;
        std::vector<int> actions;
        std::vector<double> rewards;
        std::vector<bool> dones;

        for (auto& [s, a, r, ns, d] : batch) {
            states.push_back(s);
            actions.push_back(a);
            rewards.push_back(r);
            next_states.push_back(ns);
            dones.push_back(d);
        }

        auto state_tensor = torch::stack(states);
        auto next_state_tensor = torch::stack(next_states);
        auto action_tensor = torch::tensor(actions, torch::kLong);
        auto reward_tensor = torch::tensor(rewards);
        auto done_tensor = torch::tensor(std::vector<int64_t>(dones.begin(), dones.end()), torch::kBool);


        // Compute Q values
        auto q_values = policy_net->forward(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1);
        auto next_q_output = target_net->forward(next_state_tensor).max(1);
        auto next_q_values = std::get<0>(next_q_output).detach(); // Get max values
        auto target_q_values = reward_tensor + GAMMA * next_q_values * (~done_tensor);

        // Compute loss and optimize
        auto loss = torch::mse_loss(q_values, target_q_values);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    void update_target() {
        // ✅ Save policy model
        policy_net->save_model("policy_model.pth");

        // ✅ Load into target model
        target_net->load_model("policy_model.pth");
    }

private:
    DQN policy_net;
    DQN target_net;
    torch::optim::Adam optimizer;
};
