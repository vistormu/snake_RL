from src import entities
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
    game = entities.SnakeGameAI()

    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(
            state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
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
