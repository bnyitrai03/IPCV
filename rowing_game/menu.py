# game/menu.py
import pygame, os

from rowing_game.input_schemes import BLEScheme, KeyboardScheme
from rowing_game.player import Player
from .settings import *
from .ble_message import Message

BASE_DIR = os.path.dirname(__file__)

class Menu:
    def __init__(self, ble_message: Message, start_mode: str = "menu"):
        self.ble_message = ble_message
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Canoe Game - Menu")
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self.title_font = pygame.font.SysFont("Impact", 100)
        except:
            self.title_font = pygame.font.SysFont(None, 100)
        self.font = pygame.font.SysFont("Bahnschrift", 28) if pygame.font.get_init() else pygame.font.SysFont(None, 28)
        self.small_font = pygame.font.SysFont("Bahnschrift", 22) if pygame.font.get_init() else pygame.font.SysFont(None, 22)
        # Larger font for configuration screen
        try:
            self.config_font = pygame.font.SysFont("Bahnschrift", 38)
        except:
            self.config_font = pygame.font.SysFont(None, 38)

        # Background
        menu_bg = pygame.image.load(os.path.join(BASE_DIR, 'images', 'menu.png')).convert()
        self.menu_bg = pygame.transform.scale(menu_bg, (SCREEN_WIDTH, SCREEN_HEIGHT))

        # Load settings
        self.settings = load_settings()
        self.mode = "settings" if start_mode == "settings" else "menu"

        # Music
        pygame.mixer.init()
        self.menu_music_path = os.path.join(BASE_DIR, 'sounds', 'menu.mp3')
        if os.path.exists(self.menu_music_path):
            pygame.mixer.music.load(self.menu_music_path)
            pygame.mixer.music.set_volume(self.settings.music_vol / 100)
            pygame.mixer.music.play(-1)

        # Main menu buttons
        cx = SCREEN_WIDTH//2 - 160
        self.btn_start    = (pygame.Rect(cx, 280, 320, 62), "Start")
        self.btn_settings = (pygame.Rect(cx, 355, 320, 62), "Settings")
        self.btn_quit     = (pygame.Rect(cx, 430, 320, 62), "Quit")
        self.btn_back     = (pygame.Rect(cx, 480, 320, 58), "Save & Back")

        # Settings sliders
        self.music_rect = pygame.Rect(SCREEN_WIDTH//2-220, 140, 440, 24)
        self.sfx_rect   = pygame.Rect(SCREEN_WIDTH//2-220, 200, 440, 24)

        self.players_minus = pygame.Rect(SCREEN_WIDTH//2-240, 250, 44, 44)
        self.players_plus  = pygame.Rect(SCREEN_WIDTH//2+196, 250, 44, 44)

        self.diff_left     = pygame.Rect(SCREEN_WIDTH//2-240, 300, 44, 44)
        self.diff_right    = pygame.Rect(SCREEN_WIDTH//2+196, 300, 44, 44)

        # Names section
        start_names_y = 370
        self.name_rects = [pygame.Rect(SCREEN_WIDTH//2-220, start_names_y + i*36, 440, 28) for i in range(2)]
        self.active_name = -1

    def _draw_btn(self, btn):
        rect, text = btn
        pygame.draw.rect(self.screen, BTN_BG, rect, border_radius=14)
        pygame.draw.rect(self.screen, BTN_OUT, rect, 2, border_radius=14)
        label = self.font.render(text, True, (25,25,25))
        self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_slider(self, rect, value, label):
        pygame.draw.rect(self.screen, BTN_OUT, rect, 2, border_radius=8)
        filled = pygame.Rect(rect.x, rect.y, rect.w * (value / 100), rect.h)
        pygame.draw.rect(self.screen, (40,140,255), filled, border_radius=8)
        txt = self.small_font.render(f"{label}: {int(value)}%", True, (255,255,255))
        self.screen.blit(txt, (rect.x, rect.y - 22))

    def _stop_menu_music(self):
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def configure_player(self):
        configuring = True
        selected_config = "NONE"
        player = None
        buttons = {
            "Paddles": pygame.Rect(SCREEN_WIDTH//4 - 100, 2*SCREEN_HEIGHT//3, 200, 50),
            "Keyboard": pygame.Rect(3*SCREEN_WIDTH//4 - 100, 2*SCREEN_HEIGHT//3, 200, 50),
        }
        selected_keys = []

        while configuring:
            pygame.event.pump()
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit", None
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    if buttons["Paddles"].collidepoint(mouse_pos):
                        selected_config = "PADDLES"
                    elif buttons["Keyboard"].collidepoint(mouse_pos):
                        selected_config = "KEYBOARD"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_k and selected_config == "NONE":
                        selected_config = "KEYBOARD"
                    elif event.key == pygame.K_p and selected_config == "NONE":
                        selected_config = "PADDLES"
                    elif event.key == pygame.K_RETURN and player != None:
                        return "start", player
                    elif event.key == pygame.K_q:
                        return "quit", None
                    elif selected_config == "KEYBOARD":
                        selected_keys.append(event.key)

            self.screen.blit(self.menu_bg, (0, 0))

            match(selected_config):
                case("NONE"):
                    # Create semi-transparent background box for text
                    text_box = pygame.Surface((700, 200), pygame.SRCALPHA)
                    text_box.fill((0, 0, 0, 180))
                    self.screen.blit(text_box, ((SCREEN_WIDTH - 700)//2, SCREEN_HEIGHT//3 - 60))

                    t0 = self.config_font.render("Choose how to play:", True, (255, 255, 255))
                    t1 = self.config_font.render("To play with paddles, press p.", True, (255, 255, 255))
                    t2 = self.config_font.render("To play with a keyboard, press k.", True, (255, 255, 255))
                    self.screen.blit(t0, (SCREEN_WIDTH//2 - t0.get_width()//2, SCREEN_HEIGHT//3 - 30))
                    self.screen.blit(t1, (SCREEN_WIDTH//2 - t1.get_width()//2, SCREEN_HEIGHT//3 + 20))
                    self.screen.blit(t2, (SCREEN_WIDTH//2 - t2.get_width()//2, SCREEN_HEIGHT//3 + 70))

                    # Draw buttons using new style
                    for key, rect in buttons.items():
                        pygame.draw.rect(self.screen, BTN_BG, rect, border_radius=14)
                        pygame.draw.rect(self.screen, BTN_OUT, rect, 2, border_radius=14)
                        text = self.font.render(key, True, (25,25,25))
                        text_rect = text.get_rect(center=rect.center)
                        self.screen.blit(text, text_rect)
                case("PADDLES"):
                    if player == None:
                        player = Player(BLEScheme(self.ble_message))

                    # Create semi-transparent background box for text
                    text_box = pygame.Surface((700, 220), pygame.SRCALPHA)
                    text_box.fill((0, 0, 0, 180))
                    self.screen.blit(text_box, ((SCREEN_WIDTH - 700)//2, SCREEN_HEIGHT//3 - 60))

                    t0 = self.config_font.render("Row with the left paddle to activate", True, (255, 255, 255))
                    t1 = self.config_font.render("the red circle or the right paddle", True, (255, 255, 255))
                    t2 = self.config_font.render("to activate the blue circle.", True, (255, 255, 255))
                    t3 = self.config_font.render("To continue, press ENTER.", True, (255, 255, 120))
                    self.screen.blit(t0, (SCREEN_WIDTH//2 - t0.get_width()//2, SCREEN_HEIGHT//3 - 30))
                    self.screen.blit(t1, (SCREEN_WIDTH//2 - t1.get_width()//2, SCREEN_HEIGHT//3 + 10))
                    self.screen.blit(t2, (SCREEN_WIDTH//2 - t2.get_width()//2, SCREEN_HEIGHT//3 + 50))
                    self.screen.blit(t3, (SCREEN_WIDTH//2 - t3.get_width()//2, SCREEN_HEIGHT//3 + 100))

                    left_ok  = bool(self.ble_message.LEFT)
                    right_ok = bool(self.ble_message.RIGHT)

                    # Circles: at the start they are greyish and after configuring they are red and blue
                    pygame.draw.circle(self.screen, (200, 0, 0) if left_ok else (100, 100, 100),
                                    (SCREEN_WIDTH//2 - 120, SCREEN_HEIGHT//2 + 80), 35)
                    ltxt = self.config_font.render("Left (Red)", True, (255, 255, 255))
                    self.screen.blit(ltxt, (SCREEN_WIDTH//2 - 180, SCREEN_HEIGHT//2 + 130))

                    pygame.draw.circle(self.screen, (0, 0, 200) if right_ok else (100, 100, 100),
                                    (SCREEN_WIDTH//2 + 120, SCREEN_HEIGHT//2 + 80), 35)
                    rtxt = self.config_font.render("Right (Blue)", True, (255, 255, 255))
                    self.screen.blit(rtxt, (SCREEN_WIDTH//2 + 50, SCREEN_HEIGHT//2 + 130))
                case("KEYBOARD"):
                    # Create semi-transparent background box for text
                    text_box = pygame.Surface((700, 150), pygame.SRCALPHA)
                    text_box.fill((0, 0, 0, 180))
                    self.screen.blit(text_box, ((SCREEN_WIDTH - 700)//2, SCREEN_HEIGHT//3 - 40))

                    if len(selected_keys) == 0:
                        t1 = self.config_font.render("Press a key (except for q)", True, (255, 255, 255))
                        t2 = self.config_font.render("to use as the left paddle.", True, (255, 255, 255))
                        self.screen.blit(t1, (SCREEN_WIDTH//2 - t1.get_width()//2, SCREEN_HEIGHT//3))
                        self.screen.blit(t2, (SCREEN_WIDTH//2 - t2.get_width()//2, SCREEN_HEIGHT//3 + 45))
                    elif len(selected_keys) == 1:
                        t1 = self.config_font.render("Press a key (except for q)", True, (255, 255, 255))
                        t2 = self.config_font.render("to use as the right paddle.", True, (255, 255, 255))
                        self.screen.blit(t1, (SCREEN_WIDTH//2 - t1.get_width()//2, SCREEN_HEIGHT//3))
                        self.screen.blit(t2, (SCREEN_WIDTH//2 - t2.get_width()//2, SCREEN_HEIGHT//3 + 45))
                    else:
                        if (player == None):
                            player = Player(KeyboardScheme(selected_keys[0], selected_keys[1]))
                        t1 = self.config_font.render("To continue, press ENTER.", True, (255, 255, 120))
                        self.screen.blit(t1, (SCREEN_WIDTH//2 - t1.get_width()//2, SCREEN_HEIGHT//3 + 20))

            pygame.display.flip()
            self.clock.tick(30)

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self._stop_menu_music()
                    return "quit", None
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q and self.mode == "menu":
                        self._stop_menu_music()
                        return "quit", None

                if self.mode == "menu" and e.type == pygame.MOUSEBUTTONDOWN:
                    if self.btn_start[0].collidepoint(e.pos):
                        # Go to player configuration
                        res, player = self.configure_player()
                        if res == "quit":
                            self._stop_menu_music()
                            return "quit", None
                        else:
                            self._stop_menu_music()
                            return res, player
                    if self.btn_settings[0].collidepoint(e.pos):
                        self.mode = "settings"
                    if self.btn_quit[0].collidepoint(e.pos):
                        self._stop_menu_music()
                        return "quit", None

                elif self.mode == "settings":
                    mx, my = pygame.mouse.get_pos()
                    if e.type == pygame.MOUSEBUTTONDOWN:
                        self.active_name = -1
                        if self.music_rect.collidepoint((mx,my)):
                            self.settings.music_vol = max(0, min(100, int(((mx - self.music_rect.x)/self.music_rect.w)*100)))
                            try:
                                pygame.mixer.music.set_volume(self.settings.music_vol / 100)
                            except Exception:
                                pass
                        elif self.sfx_rect.collidepoint((mx,my)):
                            self.settings.sfx_vol = max(0, min(100, int(((mx - self.sfx_rect.x)/self.sfx_rect.w)*100)))
                        elif self.players_minus.collidepoint((mx,my)):
                            self.settings.players = max(1, self.settings.players-1)
                        elif self.players_plus.collidepoint((mx,my)):
                            self.settings.players = min(2, self.settings.players+1)
                        elif self.diff_left.collidepoint((mx,my)) or self.diff_right.collidepoint((mx,my)):
                            order = ["easy","hard"]
                            i = order.index(self.settings.difficulty) if self.settings.difficulty in order else 0
                            i = (i + (1 if self.diff_right.collidepoint((mx,my)) else -1)) % len(order)
                            self.settings.difficulty = order[i]
                        elif self.btn_back[0].collidepoint(e.pos):
                            save_settings(self.settings)
                            self.mode = "menu"
                        else:
                            for i in range(self.settings.players):
                                if self.name_rects[i].collidepoint((mx,my)):
                                    self.active_name = i
                                    break
                    elif e.type == pygame.KEYDOWN and self.active_name != -1:
                        if e.key == pygame.K_BACKSPACE:
                            self.settings.player_names[self.active_name] = self.settings.player_names[self.active_name][:-1]
                        elif e.key == pygame.K_RETURN:
                            self.active_name = -1
                        else:
                            ch = e.unicode
                            if ch and ch.isprintable():
                                self.settings.player_names[self.active_name] += ch

            # Draw
            self.screen.blit(self.menu_bg, (0, 0))

            if self.mode == "menu":
                title = self.title_font.render("CANOE GAME", True, (255,255,255))
                shadow = self.title_font.render("CANOE GAME", True, (0,0,0))
                self.screen.blit(shadow, shadow.get_rect(center=(SCREEN_WIDTH//2+3, 120+3)))
                self.screen.blit(title,  title.get_rect(center=(SCREEN_WIDTH//2,   120)))
                for b in (self.btn_start, self.btn_settings, self.btn_quit):
                    self._draw_btn(b)
            else:
                title = self.title_font.render("Settings", True, (255,255,255))
                self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH//2, 80)))

                self._draw_slider(self.music_rect, self.settings.music_vol, "Music Volume")
                self._draw_slider(self.sfx_rect,   self.settings.sfx_vol,   "SFX Volume")

                # Players & Difficulty
                for r in (self.players_minus, self.players_plus, self.diff_left, self.diff_right):
                    pygame.draw.rect(self.screen, BTN_BG, r, border_radius=10)
                    pygame.draw.rect(self.screen, BTN_OUT, r, 2, border_radius=10)
                self.screen.blit(self.font.render("-", True, (25,25,25)), self.players_minus.move(12,5))
                self.screen.blit(self.font.render("+", True, (25,25,25)), self.players_plus.move(12,5))
                self.screen.blit(self.font.render("<", True, (25,25,25)), self.diff_left.move(12,5))
                self.screen.blit(self.font.render(">", True, (25,25,25)), self.diff_right.move(12,5))

                ptxt = self.font.render(f"Players: {self.settings.players}", True, (255,255,255))
                self.screen.blit(ptxt, ptxt.get_rect(center=(SCREEN_WIDTH//2, 274)))

                dtxt = self.font.render(f"Difficulty: {self.settings.difficulty.upper()}", True, (255,255,255))
                self.screen.blit(dtxt, dtxt.get_rect(center=(SCREEN_WIDTH//2, 324)))

                self.screen.blit(self.font.render("Player Names:", True, (255,255,255)),
                                 (SCREEN_WIDTH//2-220, 342))
                for i in range(self.settings.players):
                    r = self.name_rects[i]
                    pygame.draw.rect(self.screen, (230,230,230), r, border_radius=8)
                    pygame.draw.rect(self.screen, BTN_OUT, r, 2, border_radius=8)
                    name = self.settings.player_names[i].strip() or f"Player {i+1}"
                    self.screen.blit(self.small_font.render(name, True, (25,25,25)), (r.x+8, r.y+4))

                self._draw_btn(self.btn_back)

            pygame.display.flip()
            self.clock.tick(60)
