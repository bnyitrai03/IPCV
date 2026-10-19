from .game_core import Game
from .menu import Menu
from .yolo_gateway import YoloGateway
from .ble_message import Message
import logging.config
import yaml
import pygame

with open("rowing_game/log_config.yaml", "r") as f:
    config = yaml.safe_load(f)
logging.config.dictConfig(config)
logger = logging.getLogger("root")

def main():
    #create and start webcam pose tracking
    message = Message()
    gateway = YoloGateway(message)
    try:
        while True:
            menu = Menu(message)
            choice, player = menu.run()
            #message from menu quit or start so it quits and starts from main
            if choice == "quit":
                return
            elif choice == "start":
                # Get settings from menu
                settings = menu.settings
                game = Game(player, settings)

                print("=== CANOE ROWING GAME ===")
                print("Controls (webcam pose tracking):")
                print("  Pull BOTH hands = row forward (go UP)")
                print("  Pull LEFT hand  = left paddle (turn RIGHT)")
                print("  Pull RIGHT hand = right paddle (turn LEFT)")
                print("  Hands extended  = drift DOWN with current")
                print("\nGoal: Pass as many obstacles as possible to score points!")
                print("Keep rowing... If you hit the bottom, you lose!")
                print("\n  R = Restart (when game over)")
                print("  Q = Quit to Application (or Exit to Menu when game over)")
                print("open config (press w for webcam or k for keyboard, then start)")
                print("=" * 50)

                result = game.run()

                # If game returns "quit", exit entirely
                if result == "quit":
                    return
                # If game returns "menu", loop back to menu
                # (this happens when user presses E on game over screen)

    finally:
        gateway.stop()
        pygame.quit()

if __name__ == "__main__":
    main()
