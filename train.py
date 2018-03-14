"""Train the flappy bird agent."""
import argparse
import os
import torch
from game import FlappyBirdEnv
from agent import DQNAgent

def train(episodes=5000):
    env = FlappyBirdEnv(render=False)
    agent = DQNAgent()
    os.makedirs('checkpoints', exist_ok=True)
    best_score = 0
    scores = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
        score = info['score']
        scores.append(score)
        if score > best_score:
            best_score = score
            torch.save(agent.policy_net.state_dict(), 'checkpoints/best.pt')
        if (ep + 1) % 100 == 0:
            avg = sum(scores[-100:]) / min(len(scores), 100)
            print(f"Episode {ep+1}: score={score}, avg_100={avg:.1f}, best={best_score}, eps={agent.epsilon:.3f}")
    print(f"Training done. Best score: {best_score}")
# TODO: add CNN version that takes raw frames instead of state vector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5000)
    args = parser.parse_args()
    train(args.episodes)
