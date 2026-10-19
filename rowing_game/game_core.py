import pygame
import random
import os

from rowing_game.player import Player
from .settings import *
from .river import River
from .canoe import Canoe
from .obstacle import Obstacle
from .paddle_indicator import PaddleIndicator

BASE_DIR = os.path.dirname(__file__)

# Palette for end game messages
PALETTE = {
    "overlay":     (0, 0, 0, 200),
    "body":        (230, 245, 255),
    "title":       (255, 255, 120),
    "keycap_bg":   (30, 40, 90),
    "keycap_fg":   (255, 255, 255),
    "shadow":      (0, 0, 0, 120),
}

def quip_named(score: int, name: str) -> str:
    name = (name or "Player")
    if score == 0:            return f"Oops… looks like someone is drowning, {name}!"
    if 1  <= score <= 4:      return f"Okay {name}, I'll need you to row harder."
    if 5  <= score <= 10:     return f"Not bad {name}—keep the rhythm!"
    if 11 <= score <= 20:     return f"You're finding the current, {name}!"
    if 21 <= score <= 30:     return f"Nice flow {name}—stay sharp!"
    if 31 <= score <= 40:     return f"Strong paddling {name}! Nearly pro!"
    if 41 <= score <= 50:     return f"Elite rowing {name}—stone-cold!"
    return f"Legendary. You're a river tamer, {name}!"

def draw_keycap(surface, text, x, y, font):
    cap = pygame.Rect(0, 0, 36, 36); cap.center = (x, y)
    sh = pygame.Surface((cap.w+6, cap.h+6), pygame.SRCALPHA)
    sh.fill(PALETTE["shadow"]); surface.blit(sh, (cap.x+2, cap.y+2))
    pygame.draw.rect(surface, PALETTE["keycap_bg"], cap, border_radius=8)
    pygame.draw.rect(surface, (255,255,255,50), cap, 2, border_radius=8)
    lbl = font.render(text, True, PALETTE["keycap_fg"])
    surface.blit(lbl, lbl.get_rect(center=cap.center))

def wrap_lines(font, text, max_width):
    words, lines, line = text.split(), [], ""
    for w in words:
        test = (line + " " + w) if line else w
        if font.size(test)[0] <= max_width:
            line = test
        else:
            if line: lines.append(line)
            line = w
    if line: lines.append(line)
    return lines 

class Game:
    def __init__(self, player: Player, settings: UserSettings = None):
        self.player = player
        self.settings = settings if settings is not None else UserSettings()

        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Canoe Game - IoT Demo")
        self.clock = pygame.time.Clock()

        # Fonts for end game messages
        try:
            self.title_font = pygame.font.SysFont("Impact", 74)
            self.body_font  = pygame.font.SysFont("Bahnschrift", 36)
            self.small_font = pygame.font.SysFont("Bahnschrift", 26)
            self.key_font   = pygame.font.SysFont("Bahnschrift", 26)
        except:
            self.title_font = pygame.font.SysFont(None, 74)
            self.body_font  = pygame.font.SysFont(None, 36)
            self.small_font = pygame.font.SysFont(None, 26)
            self.key_font   = pygame.font.SysFont(None, 26)

        # Load images and scale to game sizes
        self.boat_img = pygame.image.load(os.path.join(BASE_DIR, 'images', 'boat.png')).convert_alpha()
        self.rock_img = pygame.image.load(os.path.join(BASE_DIR,'images','rock.png')).convert_alpha()

        # Scale to original game dimensions
        self.boat_img = pygame.transform.scale(self.boat_img, (CANOE_WIDTH, CANOE_HEIGHT))
        self.rock_img = pygame.transform.scale(self.rock_img, (OBSTACLE_WIDTH, OBSTACLE_HEIGHT))

        # Load sound effects
        pygame.mixer.init()
        self.sound_game_start = pygame.mixer.Sound(os.path.join(BASE_DIR,'sounds','game_start.mp3'))
        self.sound_game_over = pygame.mixer.Sound(os.path.join(BASE_DIR,'sounds','game_over.mp3'))
        self.sound_point = pygame.mixer.Sound(os.path.join(BASE_DIR,'sounds','point.mp3'))

        # Apply volume settings
        sfx_vol = self.settings.sfx_vol / 100
        self.sound_game_start.set_volume(sfx_vol)
        self.sound_game_over.set_volume(sfx_vol)
        self.sound_point.set_volume(sfx_vol)

        # Load background music
        pygame.mixer.music.load(os.path.join(BASE_DIR,'sounds','river_splashy.mp3'))
        pygame.mixer.music.set_volume(self.settings.music_vol / 100)

        # Apply difficulty settings
        spf, spdf = difficulty_factors(self.settings.difficulty)
        self.spawn_interval = BASE_SPAWN_INTERVAL * spf * 1000  # Convert to milliseconds
        self.obstacle_speed = BASE_OBSTACLE_SPEED * spdf

        self.reset_game()

        # Paddle indicators (moved to bottom of screen)
        self.left_indicator = PaddleIndicator(50, SCREEN_HEIGHT - 100, "LEFT")
        self.right_indicator = PaddleIndicator(SCREEN_WIDTH - 130, SCREEN_HEIGHT - 100, "RIGHT")

        # Player name
        self.player_name = self.settings.player_names[0].strip() if self.settings.player_names[0].strip() else "Player 1"

    def reset_game(self):
        self.river = River()
        self.canoe = Canoe(SCREEN_WIDTH // 2 - CANOE_WIDTH // 2,
                          SCREEN_HEIGHT // 2,  # Start at middle of screen
                          self.boat_img)
        self.obstacles = []
        self.game_over = False
        self.game_won = False
        self.last_spawn_time = pygame.time.get_ticks()

        self.player.score = 0

        # Play game start sound and start background music
        self.sound_game_start.play()
        pygame.mixer.music.set_volume(0.75)  # Set volume to 75% (25% reduction)
        pygame.mixer.music.play(-1)  # Loop background music indefinitely
        
    def spawn_obstacle(self):
        # Get river bounds at the top of the screen where obstacles spawn
        spawn_y = 0
        left_bound, right_bound = self.river.get_river_bounds_at_y(spawn_y)

        # Spawn obstacle within river bounds
        # Add some margin to keep obstacles away from the banks
        margin = 20
        left_bound += margin
        right_bound -= margin + OBSTACLE_WIDTH

        if right_bound > left_bound:
            x = random.randint(int(left_bound), int(right_bound))
            self.obstacles.append(Obstacle(x, -OBSTACLE_HEIGHT, self.rock_img))
        
    def check_collision(self):
        canoe_rect = self.canoe.get_collision_rect()  # Use smaller hitbox
        for obstacle in self.obstacles:
            if canoe_rect.colliderect(obstacle.get_rect()):
                return True
        return False
    
    def update(self, direction, left_paddle, right_paddle):
        if self.game_over or self.game_won:
            return

        # Update paddle indicators
        self.left_indicator.set_active(left_paddle)
        self.right_indicator.set_active(right_paddle)

        # Move canoe horizontally
        self.canoe.move(direction)

        # Handle vertical movement based on paddling
        is_rowing = left_paddle or right_paddle
        river_scroll_speed = 0
        if is_rowing:
            # Move UP (fighting the current) - river banks should move DOWN
            self.canoe.y -= UPSTREAM_SPEED
            # Limit how far up the canoe can go (around halfway)
            if self.canoe.y < CANOE_MAX_UP_Y:
                self.canoe.y = CANOE_MAX_UP_Y
            # Negative scroll makes river move DOWN on screen
            river_scroll_speed = -UPSTREAM_SPEED
            self.river.update(river_scroll_speed)
        else:
            # Drift DOWN (current pushing down) - river banks should move UP
            self.canoe.y += DOWNSTREAM_DRIFT
            # Positive scroll makes river move UP on screen
            river_scroll_speed = DOWNSTREAM_DRIFT
            self.river.update(river_scroll_speed)

        # Check if canoe hit the bottom of the screen (game over)
        if self.canoe.y + self.canoe.height >= SCREEN_HEIGHT:
            if not self.game_over:
                self.sound_game_over.play()
                pygame.mixer.music.stop()  # Stop background music
            self.game_over = True

        # Update obstacles with river scroll
        for obstacle in self.obstacles[:]:
            obstacle.update(river_scroll_speed)
            # Award points if obstacle passed the canoe
            if obstacle.y > self.canoe.y + self.canoe.height and not hasattr(obstacle, 'counted'):
                self.player.score += 1
                self.sound_point.play()  # Play point sound
                obstacle.counted = True  # Mark as counted so we don't count it again
            if obstacle.is_off_screen():
                self.obstacles.remove(obstacle)

        # Spawn new obstacles (using difficulty-based spawn interval)
        current_time = pygame.time.get_ticks()
        if current_time - self.last_spawn_time > self.spawn_interval:
            self.spawn_obstacle()
            self.last_spawn_time = current_time

        # Check collisions with obstacles
        if self.check_collision():
            if not self.game_over:
                self.sound_game_over.play()
                pygame.mixer.music.stop()  # Stop background music
            self.game_over = True

        # Check collision with river banks
        if self.river.check_collision(self.canoe.get_collision_rect()):
            if not self.game_over:
                self.sound_game_over.play()
                pygame.mixer.music.stop()  # Stop background music
            self.game_over = True
    
    def draw(self):
        # Draw river
        self.river.draw(self.screen)

        # Draw paddle indicators
        self.left_indicator.draw(self.screen)
        self.right_indicator.draw(self.screen)

        # Draw game objects
        self.canoe.draw(self.screen)
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        # Draw HUD score box (styled like menu buttons)
        box = pygame.Rect(10, 10, 240, 80)
        pygame.draw.rect(self.screen, BTN_BG,  box, border_radius=14)
        pygame.draw.rect(self.screen, BTN_OUT, box, 2, border_radius=14)

        # Name (smaller) at top of the box
        name_surface = self.small_font.render(self.player_name, True, (25,25,25))
        self.screen.blit(name_surface, name_surface.get_rect(center=(box.centerx, box.y + 22)))

        # Score (bigger) centered lower in the box
        score_font_hud = pygame.font.SysFont(None, 36)
        score_surface = score_font_hud.render(str(self.player.score), True, (25,25,25))
        self.screen.blit(score_surface, score_surface.get_rect(center=(box.centerx, box.y + 55)))

        # Draw webcam/pose diagnostics in the top-right corner
        debug_info = self.player.get_debug_info()
        if debug_info:
            debug_shadow = self.small_font.render(debug_info, True, (0, 0, 0))
            debug_surface = self.small_font.render(debug_info, True, (255, 255, 255))
            debug_x = SCREEN_WIDTH - debug_surface.get_width() - 10
            self.screen.blit(debug_shadow, (debug_x + 1, 11))
            self.screen.blit(debug_surface, (debug_x, 10))

        # Draw camera preview with pose detections
        preview = self.player.get_preview()
        if preview is not None:
            preview_h, preview_w = preview.shape[:2]
            preview_surface = pygame.image.frombuffer(
                preview.tobytes(), (preview_w, preview_h), "RGB")
            preview_rect = preview_surface.get_rect(topright=(SCREEN_WIDTH - 10, 40))
            pygame.draw.rect(self.screen, (0, 0, 0), preview_rect.inflate(8, 8), border_radius=4)
            self.screen.blit(preview_surface, preview_rect)
            pygame.draw.rect(self.screen, BTN_BG, preview_rect.inflate(8, 8), 2, border_radius=4)
        
        # Draw game over screen with personalized message
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(PALETTE["overlay"])
            self.screen.blit(overlay, (0, 0))

            # Get personalized quip based on score
            title_text = quip_named(self.player.score, self.player_name)

            left_margin, right_margin = 60, 60
            maxw = SCREEN_WIDTH - left_margin - right_margin

            # Headline (wrapped)
            y = 150
            for hl in wrap_lines(self.title_font, title_text, maxw):
                surf = self.title_font.render(hl, True, PALETTE["title"])
                self.screen.blit(surf, surf.get_rect(center=(SCREEN_WIDTH//2, y)))
                y += 64
            y += 12

            # Score display
            score_line = f"{self.player_name} — {self.player.score} pts"
            score_surf = self.body_font.render(score_line, True, (255,255,255))
            self.screen.blit(score_surf, score_surf.get_rect(center=(SCREEN_WIDTH//2, y)))
            y += 50

            # Keycaps near bottom
            key_y = SCREEN_HEIGHT - 64
            kx = SCREEN_WIDTH//2
            draw_keycap(self.screen, "R", kx-60, key_y, self.key_font)
            draw_keycap(self.screen, "Q", kx+60, key_y, self.key_font)
            label = self.small_font.render("Restart              Exit to Menu", True, PALETTE["body"])
            self.screen.blit(label, label.get_rect(center=(kx, key_y+30)))
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop with keyboard controls for testing"""
        running = True

        while running:
            direction = self.player.get_direction()

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and self.game_over:
                        self.reset_game()
                    elif event.key == pygame.K_q:
                        if self.game_over:
                            # During game over, Q exits to menu
                            return "menu"
                        else:
                            # During gameplay, Q quits entirely
                            return "quit"

            # Update and draw
            self.update(direction.direction_str, direction.left, direction.right)
            self.draw()
            self.clock.tick(FPS)

        return "quit"