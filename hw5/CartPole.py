import gym
import gym
import numpy as np
import torch
import torch.optim as optim
from matplotlib import animation
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from config import env_name, initial_exploration, batch_size, update_target, goal_score, log_interval, device, \
    replay_memory_capacity, lr
from memory import Memory
from model import QNet


def get_action(state, target_net, epsilon, env):
    """
    若生成的随机数小于epsilon，则随机选择一个动作
    否则使用target_net(state)输出的结果
    :return:下一步采取的动作,值为0或1
    """
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return target_net.get_action(state)


def update_target_model(evaluate_net, target_net):
    """
    将online_net中的weights赋值给target_net
    :param evaluate_net: 权重来源的网络
    :param target_net: 需要更新参数的网络
    :return: no return value
    """
    target_net.load_state_dict(evaluate_net.state_dict())

def display_frames_as_gif(frames):
    """
    将frames中的每一帧保存为gif图
    :param frames: 一个数组 保存游戏的每一帧
    :return: no return value
    """
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 5)
    anim.save("./CartPole_v1_result.gif", writer="pillow", fps = 30)



def main():
    """
    使用DQN算法训练网络
    :return: no return value
    """
    env = gym.make(env_name,render_mode='rgb_array')
    torch.manual_seed(500)

    # 获取观察空间的shape和动作空间的shape
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    # 创建evaluate_net 和 target_net
    evaluate_net = QNet(num_inputs, num_actions)
    target_net = QNet(num_inputs, num_actions)
    update_target_model(evaluate_net, target_net)

    optimizer = optim.Adam(evaluate_net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    evaluate_net.to(device)
    target_net.to(device)
    evaluate_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    running_score = 0
    steps = 0
    epsilon = 1.0
    loss = 0

    escape = False
    frames = []
    # 最多训练3000个episode
    for e in range(3000):
        done = False

        score = 0
        state = env.reset()[0]
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        # 开始一次episode,即一次游戏
        while not done:
            steps += 1
            action = get_action(state, target_net, epsilon, env)
            next_state, reward, done = env.step(action)[:3]
            # 如果应该要退出训练 将最后一次游戏保存下来
            if escape:
                frames.append(env.render())

            next_state = torch.Tensor(next_state).unsqueeze(0)

            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1
            # 将action独热编码
            action_one_hot = np.zeros(2)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

            # 经验回放 Experience Replay
            # steps<=initial_exploration(1000)的时候仅将样本存入记忆库memory
            # 之后当样本量足够之后才开始训练过程
            if steps > initial_exploration:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(batch_size)
                loss = QNet.train_model(evaluate_net, target_net, optimizer, batch)

                if steps % update_target == 0:
                    update_target_model(evaluate_net, target_net)
        if escape:
            env.close()
            display_frames_as_gif(frames)
            break

        score = score if score == 500.0 else score + 1
        # running_score是一个平滑的分数指标，用于跟踪每个episode中的平均表现。
        # 通过将running_score更新为过去分数和本轮分数的加权平均值，可以减少分数的波动性，使其更具稳定性。
        running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                e, running_score, epsilon))
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if running_score > goal_score:
            escape = True
    env.close()


if __name__ == "__main__":
    main()
