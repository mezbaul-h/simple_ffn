# Main Runnable file for the CE889 Assignment
# Project built by Lewis Veryard and Hugo Leon-Garza
from simple_ffn.settings import PROJECT_ROOT

from .GameLoop import GameLoop


def get_config():
    return {
        "SCREEN_HEIGHT": 1000,
        "SCREEN_WIDTH": 1600,
        "LANDER_IMG_PATH": PROJECT_ROOT / "lander" / "Sprites/rocket_lander.png",
        "BACKGROUND_IMG_PATH": PROJECT_ROOT / "lander" / "Sprites/BackGround.bmp",
        "FULLSCREEN": True,
        "ALL_DATA": False,
    }


config_data = get_config()


def start_game_window():
    game = GameLoop()
    game.init(config_data)
    game.main_loop(config_data)


start_game_window()
