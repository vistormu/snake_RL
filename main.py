from src import game as gm
from src import agent as ag
from core import logger
from src import plotter


def train():
    log = logger.Logger('main')

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = ag.Agent()
    game = gm.SnakeGameAI()

    while True:
        state = agent.get_state(game)

        action = agent.get_action(state)

        reward, game_over, score = game.play_step(action)
        new_state = agent.get_state(game)

        agent.train_short_memory(
            state, action, reward, new_state, game_over)

        agent.remember(state, action, reward, new_state, game_over)

        if game_over:
            game.reset()

            agent.n_games += 1

            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            log.info('Game: ', agent.n_games, ', Score: ',
                     score, ', Record: ', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plotter.plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
