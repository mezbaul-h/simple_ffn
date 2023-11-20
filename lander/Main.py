# Main Runnable file for the CE889 Assignment
# Project built by Lewis Veryard and Hugo Leon-Garza
from .GameLoop import GameLoop
from .constants import MODULE_ROOT_DIR


def get_config():
    return {
        'SCREEN_HEIGHT': 1000,
        'SCREEN_WIDTH': 1600,
        'LANDER_IMG_PATH': MODULE_ROOT_DIR / 'Sprites/rocket_lander.png',
        'BACKGROUND_IMG_PATH': MODULE_ROOT_DIR / 'Sprites/BackGround.bmp',
        'FULLSCREEN': True,
        'ALL_DATA': False,
    }


config_data = get_config()


def start_game_window():
    game = GameLoop()
    game.init(config_data)
    game.main_loop(config_data)


start_game_window()
