import enum
import pygame
from collections import namedtuple
import random
from src import constants as c
import numpy as np

Point = namedtuple('Point', 'x, y')


def square(screen, x, y, size, color, width=0):
    pygame.draw.polygon(screen, color, [(x*size, y*size),
                                        ((x+1)*size, y*size),
                                        ((x+1)*size, (y+1)*size),
                                        (x*size, (y+1)*size)], width)


class Direction(enum.Enum):
    right = enum.auto()
    left = enum.auto()
    up = enum.auto()
    down = enum.auto()


class SnakeGame:

    def __init__(self):
        self.x_dim = c.X_DIM
        self.y_dim = c.Y_DIM
        self.display = pygame.display.set_mode((c.WIDTH, c.HEIGHT))
        self.clock = pygame.time.Clock()

        # init game state
        self.direction = Direction.right

        self.head = Point(self.x_dim/2, self.y_dim/2)
        self.snake = [self.head,
                      Point(self.head.x-1, self.head.y),
                      Point(self.head.x-2, self.head.y)]

        self.score = 0
        self._place_food()

    def _place_food(self):
        x = random.randint(0, self.x_dim-1)
        y = random.randint(0, self.y_dim-1)
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if (event.key == pygame.K_a) and (self.direction is not Direction.right):
                    self.direction = Direction.left
                elif (event.key == pygame.K_d) and (self.direction is not Direction.left):
                    self.direction = Direction.right
                elif (event.key == pygame.K_w) and (self.direction is not Direction.down):
                    self.direction = Direction.up
                elif (event.key == pygame.K_s) and (self.direction is not Direction.up):
                    self.direction = Direction.down

        # 2. move
        self._move(self.direction)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(c.SPEED)
        # 6. return game over and score
        return game_over, self.score

    def _is_collision(self):
        # hits boundary
        if (self.head.x > self.x_dim-1) or (self.head.x < 0) or (self.head.y > self.y_dim-1) or (self.head.y < 0):
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(c.BLACK)

        for pt in self.snake:
            square(self.display, pt.x, pt.y, c.BLOCK_SIZE, c.GREEN)

        square(self.display, self.food.x, self.food.y, c.BLOCK_SIZE, c.RED)

        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction is Direction.right:
            x += 1
        elif direction is Direction.left:
            x -= 1
        elif direction is Direction.down:
            y += 1
        elif direction is Direction.up:
            y -= 1

        self.head = Point(x, y)


class SnakeGameAI:

    def __init__(self):
        self.x_dim = c.X_DIM
        self.y_dim = c.Y_DIM
        self.display = pygame.display.set_mode((c.WIDTH, c.HEIGHT))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.right

        self.head = Point(self.x_dim/2, self.y_dim/2)
        self.snake = [self.head,
                      Point(self.head.x-1, self.head.y),
                      Point(self.head.x-2, self.head.y)]

        self.score = 0
        self._place_food()
        self.iteration = 0

    def _place_food(self):
        x = random.randint(0, self.x_dim-1)
        y = random.randint(0, self.y_dim-1)
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            pass

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        game_over = False
        reward = 0
        if self._is_collision() or self.iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(c.SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def _is_collision(self, point=None):
        if not point:
            point = self.head

        # hits boundary
        if (point.x > self.x_dim-1) or (point.x < 0) or (point.y > self.y_dim-1) or (point.y < 0):
            return True
        # hits itself
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(c.BLACK)

        for pt in self.snake:
            square(self.display, pt.x, pt.y, c.BLOCK_SIZE, c.GREEN)

        square(self.display, self.food.x, self.food.y, c.BLOCK_SIZE, c.RED)

        pygame.display.flip()

    def _move(self, action):

        # [straight, right, left]
        clockwise = [Direction.right, Direction.down,
                     Direction.left, Direction.up]
        index = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clockwise[index]
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (index + 1) % 4
            new_direction = clockwise[next_index]
        else:
            next_index = (index - 1) % 4
            new_direction = clockwise[next_index]

        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction is Direction.right:
            x += 1
        elif self.direction is Direction.left:
            x -= 1
        elif self.direction is Direction.down:
            y += 1
        elif self.direction is Direction.up:
            y -= 1

        self.head = Point(x, y)
