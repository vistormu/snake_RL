import pygame
from src import entities


def main():
    pygame.init()

    game = entities.SnakeGame()

    # game loop
    while True:
        game_over, score = game.play_step()

        if game_over:
            break

    print('Final Score', score)

    pygame.quit()


if __name__ == '__main__':
    main()
