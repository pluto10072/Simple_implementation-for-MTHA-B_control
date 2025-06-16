import gymnasium as gym
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from DQN_net import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试函数
def test_model(model_path, label, color):
    policy_net = DQN(num_states, num_actions).to(device)
    if os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path))
        policy_net.eval()
        print(f"Loaded model: {model_path}")
    else:
        print(f"Model not found: {model_path}")
        return []

    rewards = []
    for ep in range(100):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_arr = np.array(state, dtype=np.float32)
            state_tensor = torch.from_numpy(state_arr).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        print(f"{label} Episode {ep+1}: Reward = {total_reward}")
    return rewards

# 环境参数
env = gym.make('LunarLander-v3')
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

# 测试两个模型
rewards1 = test_model("best_lunar_dqn_3.pth", "DDQN", "blue")
rewards2 = test_model("human_trade_dqn.pth", "HumanTrade", "orange")

# 画图
window_size = 20
def moving_avg(data, w):
    return [np.mean(data[max(0, i-w+1):i+1]) for i in range(len(data))]

plt.figure(figsize=(12, 6))
plt.plot(rewards1, color='skyblue', alpha=0.4, linewidth=0.8, label='DQN Episode Reward')
plt.plot(moving_avg(rewards1, window_size), color='blue', linewidth=2, label=f'DQN Moving Avg ({window_size})')
plt.plot(rewards2, color='orange', alpha=0.4, linewidth=0.8, label='HumanTrade Episode Reward')
plt.plot(moving_avg(rewards2, window_size), color='darkorange', linewidth=2, label=f'HumanTrade Moving Avg ({window_size})')
plt.axhline(y=200, color='red', linestyle='--', linewidth=1, label='Target Score')
plt.title('LunarLander-v3 Model Comparison', fontsize=14, pad=20)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend(loc='upper left', frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()