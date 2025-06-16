import gymnasium as gym
import pygame
import numpy as np

# 初始化 Pygame
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Lunar Lander - Keyboard Control")

# 创建 Lunar Lander 环境
env = gym.make('LunarLander-v3', render_mode='rgb_array')
state, _ = env.reset()

total_reward = 0
running = True
clock = pygame.time.Clock()


# 将图像从 gym 转换为 Pygame 格式
def gym_to_pygame(img):
    return pygame.surfarray.make_surface(np.rot90(img))


while running:
    clock.tick(30)  # 控制帧率

    # 获取键盘动作
    action = 0  # 默认不动
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action = 1
    elif keys[pygame.K_RIGHT]:
        action = 3
    elif keys[pygame.K_UP]:
        action = 2

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 执行动作
    next_state, reward, done, trunc, info = env.step(action)
    total_reward += reward

    # 渲染环境到图像
    img = env.render()
    surf = gym_to_pygame(img)
    surf = pygame.transform.scale(surf, (600, 400))  # 缩放以适应窗口
    screen.blit(surf, (0, 0))

    # 显示当前总奖励
    font = pygame.font.SysFont(None, 28)
    reward_text = font.render(f"Total Reward: {total_reward:.2f}", True, (255, 255, 255))
    screen.blit(reward_text, (10, 10))

    pygame.display.flip()

    if done or trunc:
        print(f"Episode finished with total reward: {total_reward:.2f}")
        pygame.time.wait(2000)
        state, _ = env.reset()
        total_reward = 0

env.close()
pygame.quit()
