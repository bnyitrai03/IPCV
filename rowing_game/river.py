import os
import pygame 
import random
import math
from .settings import *


BASE_DIR = os.path.dirname(__file__) 

class River:
    """Generates a curved river with S-shaped variation"""
    def __init__(self):
        self.segments = []
        self.segment_height = 5  # Smaller segments for smoother curves
        self.scroll_offset = 0
        self.total_scroll = 0  # Never wraps - accumulates forever for water animation
        self.center_offset = SCREEN_WIDTH // 2
        self.curve_phase = 0  # Phase for sinusoidal curves
        self.current_scroll_speed = 0  # Track current river scroll speed

        # Load textures and scale them to tile size
        WATER_TILE_SIZE = 128  # Size for water tiles
        GRASS_TILE_SIZE = 192  # Slightly larger for grass tiles

        water_original = pygame.image.load(os.path.join(BASE_DIR,'images','water.png')).convert_alpha()
        grass_original = pygame.image.load(os.path.join(BASE_DIR,'images','grass.png')).convert_alpha()

        # Scale down to tileable size
        self.water_texture = pygame.transform.scale(water_original, (WATER_TILE_SIZE, WATER_TILE_SIZE))
        self.grass_texture = pygame.transform.scale(grass_original, (GRASS_TILE_SIZE, GRASS_TILE_SIZE))

        # Get texture dimensions
        self.water_texture_width = WATER_TILE_SIZE
        self.water_texture_height = WATER_TILE_SIZE
        self.grass_texture_width = GRASS_TILE_SIZE
        self.grass_texture_height = GRASS_TILE_SIZE

        # Generate initial river segments
        for i in range(SCREEN_HEIGHT // self.segment_height + 20):
            self.add_segment()

    def add_segment(self):
        """Add a new river segment with smooth S-curve pattern"""
        if len(self.segments) == 0:
            offset = 0
        else:
            # Create smooth S-curve using sine wave with less randomness
            self.curve_phase += 0.02  # Smaller increment = smoother, wider curves
            sine_offset = math.sin(self.curve_phase) * 100  # Amplitude of 100
            random_variation = random.uniform(-5, 5)  # Less randomness for smoother curves
            offset = sine_offset + random_variation

            # Keep the river from going too far off center
            max_offset = 120
            offset = max(min(offset, max_offset), -max_offset)

        self.segments.append(offset)

    def update(self, speed):
        """Scroll the river"""
        self.current_scroll_speed = speed  # Store for water animation adjustment
        self.scroll_offset += speed
        self.total_scroll += speed  # Accumulate forever, never wraps

        # Scrolling forward (river moves down on screen)
        while self.scroll_offset >= self.segment_height:
            self.scroll_offset -= self.segment_height
            self.segments.pop(0)
            self.add_segment()

        # Scrolling backward (river moves up on screen)
        while self.scroll_offset < 0:
            self.scroll_offset += self.segment_height
            # Add segment at the beginning - continue the S-curve pattern
            self.curve_phase -= 0.02  # Go backwards in the curve
            sine_offset = math.sin(self.curve_phase) * 100
            random_variation = random.uniform(-5, 5)
            offset = sine_offset + random_variation
            max_offset = 120
            offset = max(min(offset, max_offset), -max_offset)

            self.segments.insert(0, offset)
            # Remove from end to keep list size reasonable
            if len(self.segments) > 200:
                self.segments.pop()

    def get_river_bounds_at_y(self, y):
        """Get the left and right bounds of the river at a given y position"""
        # Determine which segment this y position is in
        segment_index = int((y + self.scroll_offset) // self.segment_height)

        if segment_index < 0 or segment_index >= len(self.segments):
            offset = 0
        else:
            offset = self.segments[segment_index]

        center = self.center_offset + offset
        left_bound = center - RIVER_WIDTH // 2
        right_bound = center + RIVER_WIDTH // 2

        return left_bound, right_bound

    def draw(self, screen):
        """Draw the river with curved banks"""
        # Water flow with partial compensation to reduce extremes
        # NOTE: Good baseline values that were close: base // 10, compensation * 0.5
        #       (good speed when drifting/moving down, slightly slow when paddling/moving up)
        base_speed = pygame.time.get_ticks() // 10  # Faster base speed
        speed_compensation = int(self.total_scroll * 0.6)  # Slightly stronger compensation for more speed when paddling up
        water_offset = (base_speed - speed_compensation) % self.water_texture_height

        # Draw water texture
        for tile_y in range(-self.water_texture_height, SCREEN_HEIGHT + self.water_texture_height, self.water_texture_height):
            for tile_x in range(0, SCREEN_WIDTH, self.water_texture_width):
                screen.blit(self.water_texture, (tile_x, tile_y + water_offset))

        # Draw river banks with grass texture
        for i in range(len(self.segments) - 1):
            # Negate scroll_offset: positive scroll moves river DOWN, negative moves UP
            y1 = i * self.segment_height - self.scroll_offset
            y2 = (i + 1) * self.segment_height - self.scroll_offset

            if y2 < -50 or y1 > SCREEN_HEIGHT + 50:
                continue

            offset1 = self.segments[i]
            offset2 = self.segments[i + 1]

            center1 = self.center_offset + offset1
            center2 = self.center_offset + offset2

            # Left bank
            left1 = center1 - RIVER_WIDTH // 2
            left2 = center2 - RIVER_WIDTH // 2

            # Draw left shore with grass texture
            if left1 > 0:
                # Create clipping rect for left bank
                bank_rect = pygame.Rect(0, int(y1), int(left1), max(1, int(y2 - y1)))
                screen.set_clip(bank_rect)

                # Tile grass texture - use total_scroll to avoid wrapping glitches
                grass_offset = int(-self.total_scroll) % self.grass_texture_height
                for grass_y in range(-self.grass_texture_height, SCREEN_HEIGHT + self.grass_texture_height, self.grass_texture_height):
                    for grass_x in range(0, int(left1) + self.grass_texture_width, self.grass_texture_width):
                        screen.blit(self.grass_texture, (grass_x, grass_y + grass_offset))

                # Remove clipping
                screen.set_clip(None)
                # Draw bank edge (reduced width)
                pygame.draw.line(screen, BROWN, (left1, y1), (left2, y2), 2)

            # Right bank
            right1 = center1 + RIVER_WIDTH // 2
            right2 = center2 + RIVER_WIDTH // 2

            # Draw right shore with grass texture
            if right1 < SCREEN_WIDTH:
                # Create clipping rect for right bank
                bank_rect = pygame.Rect(int(right1), int(y1), SCREEN_WIDTH - int(right1), max(1, int(y2 - y1)))
                screen.set_clip(bank_rect)

                # Tile grass texture - use total_scroll to avoid wrapping glitches
                grass_offset = int(-self.total_scroll) % self.grass_texture_height
                for grass_y in range(-self.grass_texture_height, SCREEN_HEIGHT + self.grass_texture_height, self.grass_texture_height):
                    for grass_x in range(int(right1) - self.grass_texture_width, SCREEN_WIDTH + self.grass_texture_width, self.grass_texture_width):
                        screen.blit(self.grass_texture, (grass_x, grass_y + grass_offset))

                # Remove clipping
                screen.set_clip(None)
                # Draw bank edge (reduced width)
                pygame.draw.line(screen, BROWN, (right1, y1), (right2, y2), 2)

    def check_collision(self, canoe_rect):
        """Check if canoe collides with river banks"""
        canoe_center_y = canoe_rect.centery
        left_bound, right_bound = self.get_river_bounds_at_y(canoe_center_y)

        # Check if canoe is outside river bounds
        if canoe_rect.left < left_bound or canoe_rect.right > right_bound:
            return True
        return False
