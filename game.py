
import pygame
import random

from enum import Enum
from collections import namedtuple

pygame.init()

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

BLOCK_SIZE = 20
SPEED = 200

Point = namedtuple('Point', 'x y')


class Direction(Enum):

    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeGameAI:

    def __init__(self, w=600, h=400):

        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((w, h+100))

        pygame.display.set_caption(
            "AI Snake Game - Improved DQN"
        )

        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):

        self.direction = Direction.RIGHT

        self.head = Point(
            self.w // 2,
            self.h // 2
        )

        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]

        self.score = 0

        self.food = None

        self._place_food()

        self.frame_iteration = 0

        self.old_distance = self._get_distance()

    def _place_food(self):

        while True:

            x = random.randint(
                0, (self.w - BLOCK_SIZE) // BLOCK_SIZE
            ) * BLOCK_SIZE

            y = random.randint(
                0, (self.h - BLOCK_SIZE) // BLOCK_SIZE
            ) * BLOCK_SIZE

            self.food = Point(x, y)

            if self.food not in self.snake:
                break

    def _get_distance(self):

        return (
            abs(self.head.x - self.food.x) +
            abs(self.head.y - self.food.y)
        )

    def play_step(self, action):

        self.frame_iteration += 1

        # Handle closing the game window
        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                pygame.quit()
                quit()

        # Move snake
        self._move(action)

        self.snake.insert(0, self.head)

        reward = 0
        game_over = False

        # Collision or timeout
        if (
            self.is_collision()
            or self.frame_iteration > 100 * len(self.snake)
        ):

            game_over = True
            reward = -10

            return reward, game_over, self.score

        # Food eaten
        if self.head == self.food:

            self.score += 1

            reward = 10

            self._place_food()

            self.old_distance = self._get_distance()

        else:

            self.snake.pop()

            # Distance-based reward
            new_distance = self._get_distance()

            if new_distance < self.old_distance:

                reward = 0.1

            elif new_distance > self.old_distance:

                reward = -0.1

            self.old_distance = new_distance

        # Update screen
        self._update_ui()

        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_collision(self, pt=None):

        if pt is None:
            pt = self.head

        # Wall collision
        if (
            pt.x < 0 or pt.x >= self.w or
            pt.y < 0 or pt.y >= self.h
        ):

            return True

        # Body collision
        if pt in self.snake[1:]:

            return True

        return False

    def _move(self, action):

        # Actions: straight, right, left

        clock_wise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP
        ]

        idx = clock_wise.index(self.direction)

        if action == [1, 0, 0]:

            new_direction = clock_wise[idx]

        elif action == [0, 1, 0]:

            new_direction = clock_wise[(idx + 1) % 4]

        elif action == [0, 0, 1]:

            new_direction = clock_wise[(idx - 1) % 4]

        else:

            raise ValueError("Invalid action")

        self.direction = new_direction

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE

        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE

        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)

    def _update_ui(self):

        self.display.fill(BLACK, (0, 0, self.w, self.h))

        for point in self.snake:

            pygame.draw.rect(
                self.display,
                GREEN,
                pygame.Rect(
                    point.x,
                    point.y,
                    BLOCK_SIZE,
                    BLOCK_SIZE
                )
            )

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(
                self.food.x,
                self.food.y,
                BLOCK_SIZE,
                BLOCK_SIZE
            )
        )

        text = pygame.font.SysFont(
            'Arial', 25
        ).render(
            "Score: " + str(self.score),
            True,
            WHITE
        )

        self.display.blit(text, [0, 0])

        pygame.display.flip()

    def get_state(self):

        head = self.head

        point_l = Point(
            head.x - BLOCK_SIZE, head.y
        )

        point_r = Point(
            head.x + BLOCK_SIZE, head.y
        )

        point_u = Point(
            head.x, head.y - BLOCK_SIZE
        )

        point_d = Point(
            head.x, head.y + BLOCK_SIZE
        )

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [

            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food position
            self.food.x < head.x,
            self.food.x > head.x,
            self.food.y < head.y,
            self.food.y > head.y
        ]

        return [int(x) for x in state]

    def close(self):

        pygame.quit()
