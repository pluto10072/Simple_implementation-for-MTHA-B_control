import gymnasium as gym
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from DQN_net import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建环境
env = gym.make('LunarLander-v3')
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

policy_net = DQN(num_states, num_actions).to(device)

param_num = 3  # 训练模型编号
param_path = f"best_lunar_dqn_3.pth"
if os.path.exists(param_path):
    print(f"检测到编号为{param_num}的模型参数")
    policy_net.load_state_dict(torch.load(param_path))
else:
    print(f"未找到编号为{param_num}的模型参数，将从头测试。")

policy_net.eval()

score_hist = []
target_score = 200  # LunarLander-v3 通常以200为目标分数

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
    score_hist.append(total_reward)
    print(f"Episode {ep+1}: Reward = {total_reward}")

# 计算滑动平均奖励（窗口大小为100）
window_size = 100
avg_scores = [np.mean(score_hist[max(0, i-window_size+1):i+1])
              for i in range(len(score_hist))]

plt.figure(figsize=(12, 6))

# 绘制单局奖励（浅色）
plt.plot(score_hist,
         color='skyblue',
         alpha=0.4,
         linewidth=0.8,
         label='Episode Reward')

# 绘制滑动平均（深色）
plt.plot(avg_scores,
         color='darkblue',
         linewidth=2,
         label=f'Moving Average ({window_size} episodes)')

# 标注目标线
plt.axhline(y=target_score,
            color='red',
            linestyle='--',
            linewidth=1,
            label='Target Score')

# 美化图表
plt.title('LunarLander DQN Test MOA Progress', fontsize=14, pad=20)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend(loc='upper left', frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存高清图片
plt.savefig('test_progress.png', dpi=300)
plt.show()