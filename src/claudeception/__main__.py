import trio

from claudeception import GameSetup


def main():
    game = GameSetup.random(6).init_game().run_night_phase()
    for _ in range(10):
        trio.run(game.collect_samples)
    trio.run(game.end_game)


if __name__ == "__main__":
    main()
