
import sys
import pygame
import torch

from game import SnakeGameAI
from agent import Agent

# Initialize Pygame
pygame.init()

# Load trained model
agent = Agent()

agent.model.load_state_dict(
    torch.load(
        "models/improved_best_model.pth",
        map_location="cpu"
    )
)

agent.model.eval()

# Game settings
WIDTH = 600
HEIGHT = 400
PANEL_HEIGHT = 100

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT + PANEL_HEIGHT)
)

pygame.display.set_caption("AI Snake Game - DQN")

# Create game
game = SnakeGameAI(WIDTH, HEIGHT)

# Fonts
font = pygame.font.SysFont("Arial", 20, bold=True)
big_font = pygame.font.SysFont("Arial", 40, bold=True)
score_font = pygame.font.SysFont("Arial", 25, bold=True)

clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (40, 170, 70)
BLUE = (50, 120, 210)
ORANGE = (220, 150, 30)
RED = (210, 50, 50)
GREY = (70, 70, 70)

# Game status
status = "ready"
final_score = 0

# Buttons
play_button = pygame.Rect(25, 425, 115, 50)
pause_button = pygame.Rect(165, 425, 115, 50)
restart_button = pygame.Rect(305, 425, 125, 50)
exit_button = pygame.Rect(455, 425, 115, 50)

center_button = pygame.Rect(
    WIDTH // 2 - 90,
    HEIGHT // 2 + 35,
    180,
    50
)


def draw_button(rect, text, color):

    pygame.draw.rect(
        screen,
        color,
        rect,
        border_radius=10
    )

    pygame.draw.rect(
        screen,
        WHITE,
        rect,
        2,
        border_radius=10
    )

    label = font.render(text, True, WHITE)

    screen.blit(
        label,
        label.get_rect(center=rect.center)
    )


def draw_controls():

    # Control panel
    pygame.draw.rect(
        screen,
        GREY,
        (0, HEIGHT, WIDTH, PANEL_HEIGHT)
    )

    draw_button(play_button, "PLAY", GREEN)

    if status == "paused":
        draw_button(pause_button, "RESUME", GREEN)
    else:
        draw_button(pause_button, "PAUSE", ORANGE)

    draw_button(restart_button, "RESTART", BLUE)
    draw_button(exit_button, "EXIT", RED)


def draw_overlay(title, subtitle, button_text):

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))

    screen.blit(overlay, (0, 0))

    title_text = big_font.render(
        title, True, WHITE
    )

    screen.blit(
        title_text,
        title_text.get_rect(
            center=(WIDTH // 2, HEIGHT // 2 - 55)
        )
    )

    subtitle_text = font.render(
        subtitle, True, WHITE
    )

    screen.blit(
        subtitle_text,
        subtitle_text.get_rect(
            center=(WIDTH // 2, HEIGHT // 2)
        )
    )

    draw_button(
        center_button,
        button_text,
        GREEN
    )


def restart_game():

    global status, final_score

    game.reset()

    final_score = 0
    status = "running"


print("AI Snake Game Started!")

# Main loop
while True:

    # Process events
    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            game.close()
            sys.exit()

        
        if event.type == pygame.MOUSEBUTTONDOWN:

            pos = event.pos

            # Exit button
            if exit_button.collidepoint(pos):
                game.close()
                sys.exit()

            # Restart button
            elif restart_button.collidepoint(pos):
                restart_game()

            # Pause / Resume button
            elif pause_button.collidepoint(pos):

                if status == "running":
                    status = "paused"

                elif status == "paused":
                    status = "running"

            # Play button
            elif play_button.collidepoint(pos):

                if status == "ready":
                    status = "running"

                elif status == "paused":
                    status = "running"

            # Click anywhere on the GAME AREA
            elif pos[1] < HEIGHT:

                if status == "running":
                    status = "paused"

                elif status == "paused":
                    status = "running"

                elif status == "ready":
                    status = "running"

                elif status == "over":
                    restart_game()


        # Keyboard shortcuts
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_SPACE:

                if status == "running":
                    status = "paused"

                elif status == "paused":
                    status = "running"

            elif event.key == pygame.K_r:

                restart_game()

            elif event.key == pygame.K_RETURN:

                if status in ("ready", "over"):
                    restart_game()

    # AI movement
    if status == "running":

        state = game.get_state()

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():

            prediction = agent.model(state_tensor)

        move = torch.argmax(prediction).item()

        action = [0, 0, 0]
        action[move] = 1

        reward, done, score = game.play_step(action)

        if done:

            final_score = score
            status = "over"

            print("Game Over! Final Score:", final_score)

    # Draw controls after game redraw
    draw_controls()

    # Ready screen
    if status == "ready":

        draw_overlay(
            "AI SNAKE GAME",
            "Click PLAY to start",
            "PLAY"
        )

    # Pause screen
    elif status == "paused":

        draw_overlay(
            "GAME PAUSED",
            "Click RESUME to continue",
            "RESUME"
        )

    # Game over screen
    elif status == "over":

        draw_overlay(
            "GAME OVER!",
            "Final Score: " + str(final_score),
            "PLAY AGAIN"
        )

    pygame.display.flip()

    # Prevent unnecessary CPU usage when not playing
    if status != "running":
        clock.tick(30)
