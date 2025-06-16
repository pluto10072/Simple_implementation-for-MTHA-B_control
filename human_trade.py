import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('LunarLander-v3')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

    # 带 dropout 的前向传播（用于 MC Dropout 估计不确定性）
    def mc_dropout_forward(self, x, T=10):
        self.train()
        outputs = torch.stack([self.forward(x) for _ in range(T)], dim=2)
        return outputs.mean(dim=2), outputs.var(dim=2)

def train_dqn(env, episodes=500, gamma=0.99, lr=1e-3):
    policy_net = DQN(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    rewards = []

    for ep in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            # 更新网络（简化为 TD update）
            target = reward + gamma * policy_net(torch.FloatTensor(next_state).unsqueeze(0).to(device)).max().item() * (
                not done)
            current_q = policy_net(state_tensor)[0, action]
            loss = loss_fn(current_q, torch.tensor(target).float().to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        rewards.append(total_reward)
        print(f"Episode {ep + 1}, Reward: {total_reward}")

    return policy_net, rewards


# 人类行为模拟（简单规则-based）
def human_action(state):
    y_velocity = state[1]
    angle = state[4]

    if y_velocity < -0.5:
        return 2  # 开主引擎
    elif angle > 0.1:
        return 3  # 向右喷射
    elif angle < -0.1:
        return 1  # 向左喷射
    else:
        return 0  # 不动作


# 可信度评估（MC Dropout）
def compute_credibility(model, state, T=10):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        mean, var = model.mc_dropout_forward(state_tensor, T=T)
    return var.mean().item()


# 自主边界计算（简化）
def compute_autonomous_boundary(model, state):
    # 这里简化为当前状态下的最大 Q 值作为边界参考
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
    return q_values.max().item()


# 仲裁机制（MTHA-B）
def arbitration_mtha_b(machine_q, human_a, machine_a, boundary, cm, ch):
    if cm >= ch and machine_q >= max(human_a, boundary):
        return 'machine', machine_a
    elif ch >= cm and human_a >= max(machine_q, boundary):
        return 'human', human_a
    else:
        return 'boundary', boundary


# 主程序：运行 MTHA-B 控制
def run_traded_control(env, policy_model, episodes=100):
    results = []

    for ep in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            # 机器决策
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_model(state_tensor)
                machine_action = q_values.argmax().item()
                machine_q_value = q_values[0, machine_action].item()

            # 人类决策
            human_action_ = human_action(state)

            # 自主边界
            boundary = compute_autonomous_boundary(policy_model, state)

            # 可信度评估
            cm = compute_credibility(policy_model, state)
            ch = compute_credibility(policy_model, state)  # 假设人类行为可信度相同（可自定义）

            # 仲裁机制
            decision_type, final_action = arbitration_mtha_b(machine_q_value, human_action_, machine_action, boundary,
                                                             cm, ch)

            next_state, reward, done, _, _ = env.step(final_action)
            total_reward += reward
            state = next_state

        results.append(total_reward)
        print(f"[MTHA-B] Episode {ep + 1}, Total Reward: {total_reward}")

    return results


# 可视化奖励曲线
def plot_rewards(rewards, title="Reward Curve"):
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("Training DQN...")
    dqn_model, dqn_rewards = train_dqn(env, episodes=200)

    torch.save(dqn_model.state_dict(), "human_trade_dqn.pth")
    print("模型参数已保存为 human_trade_dqn.pth")

    print("\nRunning MTHA-B control...")
    traded_rewards = run_traded_control(env, dqn_model, episodes=100)

    plot_rewards(dqn_rewards, "DQN Training Rewards")
    plot_rewards(traded_rewards, "MTHA-B Trading Control Rewards")