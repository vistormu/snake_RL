import torch
import random
import numpy as np
from collections import deque
from src import constants as c
from src import entities
from src import model


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=c.MAX_MEMORY)
        self.model = model.LinearQNet(
            c.INPUT_SIZE, c.HIDDEN_SIZE, c.OUTPUT_SIZE)
        self.trainer = model.QTrainer(self.model, lr=c.LR, gamma=self.gamma)

    def get_state(self, game: entities.SnakeGameAI):
        head = game.snake[0]

        point_left = entities.Point(head.x-1, head.y)
        point_right = entities.Point(head.x+1, head.y)
        point_up = entities.Point(head.x, head.y-1)
        point_down = entities.Point(head.x, head.y+1)

        dir_left = game.direction == entities.Direction.left
        dir_right = game.direction == entities.Direction.right
        dir_up = game.direction == entities.Direction.up
        dir_down = game.direction == entities.Direction.down

        def _get_danger(direction):
            return (direction[0] and game.is_collision(point_right)) or \
                (direction[1] and game.is_collision(point_left)) or  \
                (direction[2] and game.is_collision(point_up)) or \
                (direction[3] and game.is_collision(point_down))

        state = [
            # Danger straight
            _get_danger([dir_right, dir_left, dir_up, dir_down]),

            # Danger right
            _get_danger([dir_up, dir_down, dir_left, dir_right]),

            # Danger left
            _get_danger([dir_down, dir_up, dir_right, dir_left]),

            # Move direction
            dir_left,
            dir_right,
            dir_up,
            dir_down,

            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > c.BATCH_SIZE:
            mini_sample = random.sample(self.memory, c.BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
