"""Flappy Bird environment using pygame."""
import numpy as np
try:
    import pygame
except ImportError:
    pygame = None

class FlappyBirdEnv:
    SCREEN_W = 288
    SCREEN_H = 512
    PIPE_GAP = 120
    GRAVITY = 1
    FLAP_SPEED = -9
    PIPE_SPEED = -4

    def __init__(self, render=False):
        self.render_mode = render
        if render and pygame:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H))
        self.reset()

    def reset(self):
        self.bird_y = self.SCREEN_H // 2
        self.bird_vel = 0
        self.pipes = []
        self.score = 0
        self.frame = 0
        self._spawn_pipe()
        return self._get_state()

    def _spawn_pipe(self):
        gap_y = np.random.randint(100, self.SCREEN_H - 100)
        self.pipes.append({'x': self.SCREEN_W, 'gap_y': gap_y})

    def step(self, action):
        if action == 1:
            self.bird_vel = self.FLAP_SPEED
        self.bird_vel += self.GRAVITY
        self.bird_y += self.bird_vel
        for p in self.pipes:
            p['x'] += self.PIPE_SPEED
        for p in self.pipes:
            if p['x'] + 52 < 50 and not p.get('scored'):
                self.score += 1
                p['scored'] = True
        if not self.pipes or self.pipes[-1]['x'] < self.SCREEN_W - 200:
            self._spawn_pipe()
        self.pipes = [p for p in self.pipes if p['x'] > -60]

        done = False
        if self.bird_y < 0 or self.bird_y > self.SCREEN_H:
            done = True
        for p in self.pipes:
            if 50 < p['x'] < 100:
                if self.bird_y < p['gap_y'] - self.PIPE_GAP // 2 or self.bird_y > p['gap_y'] + self.PIPE_GAP // 2:
                    done = True

        reward = -1.0 if done else 0.1
        self.frame += 1
        return self._get_state(), reward, done, {'score': self.score}

    def _get_state(self):
        state = np.zeros(4, dtype=np.float32)
        state[0] = self.bird_y / self.SCREEN_H
        state[1] = self.bird_vel / 20.0
        if self.pipes:
            state[2] = self.pipes[0]['x'] / self.SCREEN_W
            state[3] = self.pipes[0]['gap_y'] / self.SCREEN_H
        return state
