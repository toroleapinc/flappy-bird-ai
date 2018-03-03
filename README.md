# Flappy Bird DQN

Teaching an AI to play Flappy Bird with Deep Q-Networks. Uses a dueling DQN architecture with prioritized experience replay.

The bird learns from a simplified state vector (position, velocity, pipe distance). There's a TODO to switch to raw pixel input with CNN.

## Setup
```
pip install -r requirements.txt
python train.py
python play.py --model checkpoints/best.pt
```

Needs pygame for the environment.
