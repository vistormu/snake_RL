import pygame
from src import entities
from core import logger
import numpy as np


def main():
    log = logger.Logger('tmp')
    log.init()

    pygame.init()

    game = entities.SnakeGameAI()

    actions = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])

    # game loop
    while True:
        for action in actions:
            reward, game_over, score = game.play_step(action)

            if game_over:
                break

        if game_over:
            break

    print('Final Score', score)

    pygame.quit()


if __name__ == '__main__':
    main()
