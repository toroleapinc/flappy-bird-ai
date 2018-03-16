"""Watch the trained agent play."""
import argparse
import torch
from game import FlappyBirdEnv
from dqn import DuelingDQN

def play(model_path, episodes=10):
    env = FlappyBirdEnv(render=True)
    model = DuelingDQN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = model(torch.FloatTensor(state).unsqueeze(0)).argmax(dim=1).item()
            state, _, done, info = env.step(action)
        print(f"Episode {ep+1}: score = {info['score']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()
    play(args.model, args.episodes)
