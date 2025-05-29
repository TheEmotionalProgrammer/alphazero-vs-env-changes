from typing import Optional, Union

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from numba import njit
from pygame import gfxdraw

PI = np.pi
DT = 0.25
SPEED_LIMIT = 4.0
STEERING_LIMIT = PI / 4
ACC_LIMIT = 1.0


CAR_LENGTH = 4.8
CAR_WIDTH = 1.8

N_SAMPLES_ACC = 2  # e.g., backward, coast, forward
N_SAMPLES_STEER = 5  # same as before

ACC_ACTIONS = np.linspace(-ACC_LIMIT, ACC_LIMIT, N_SAMPLES_ACC)
STEERING_ACTIONS = np.linspace(-STEERING_LIMIT, STEERING_LIMIT, N_SAMPLES_STEER)

STATE_H = 128
STATE_W = 128
SCREEN_H = 400
SCREEN_W = 500

SCALE = 8
STATE_SCALE = np.array(
    [SCREEN_W / SCALE, SCREEN_H / SCALE, 1, 1, SPEED_LIMIT]
    + [SCREEN_W / SCALE, SCREEN_H / SCALE] * 4
)
FPS = 30
# Modern soft color palette
RED = (239, 83, 80)       # Soft red
GREEN = (76, 175, 80)     # Emerald green
BLUE = (66, 165, 245)     # Sky blue
YELLOW = (255, 235, 59)   # Warm yellow
BLACK = (33, 33, 33)      # Charcoal black
GREY = (120, 144, 156)    # Cool grey
WHITE = (245, 245, 245)   # Off-white
PURPLE = (171, 71, 188)   # Optional for accents
TEAL = (0, 150, 136)      # Optional contrast
GOLDEN = (255, 193, 7)      # Optional gold for highlights

#BACKGROUND_COLOR = (38, 50, 56)  # a more dramatic dark theme # light modern background

# Create a light gray background color
BACKGROUND_COLOR = (100, 100, 100)  

# position[2], heading, speed, length, width, type
FIXED_INIT_STATE = np.array([-18, 18, -PI/2, 0, CAR_LENGTH, CAR_WIDTH, 0])
RAND_INIT_STATE = np.array([0, 10, -PI, 0, CAR_LENGTH, CAR_WIDTH, 0])

GOAL_STATE = np.array([20.0, -18, 0, 0, 2, 2, 1])

# Define wall shapes separately
WALLS = np.array([
    [0, (SCREEN_H - 3) / SCALE / 2, 0, 0, SCREEN_W / SCALE, 1, 1],
    [0, -SCREEN_H / SCALE / 2, 0, 0, SCREEN_W / SCALE, 1, 1],
    [(SCREEN_W - 3) / SCALE / 2, 0, 0, 0, 1, SCREEN_H / SCALE, 1],
    [(-SCREEN_W + 3) / SCALE / 2, 0, 0, 0, 1, SCREEN_H / SCALE, 1],
    [-SCREEN_W / SCALE / 2, 0, 0, 0, 1, SCREEN_H / SCALE, 1],
])

# Default obstacles (stationary vehicles)
DEFAULT_OBSTACLES = np.array([
    [-5 * 3 / 2, -20, PI / 2, 0, CAR_LENGTH, CAR_WIDTH, 0],
    [-3 * 3 / 2, -20, PI / 2, 0, CAR_LENGTH, CAR_WIDTH, 0],
    [-5 * 3 / 2, -5, -PI / 2, 0, CAR_LENGTH, CAR_WIDTH, 0],
    [7 * 3 / 2, -5, -PI / 2, 0, CAR_LENGTH, CAR_WIDTH, 0],
    [-3 / 2, 5, PI / 2, 0, CAR_LENGTH, CAR_WIDTH, 0],
    [-3 * 3 / 2, 20, -PI / 2, 0, CAR_LENGTH, CAR_WIDTH, 0],
    [-3 * 3 / 2, 0, -PI / 2, 0, CAR_LENGTH, CAR_WIDTH, 0],
])

MAX_STEPS = 256

@njit("types.none(f8[:], f8[:], f8)", fastmath=True, cache=True)
def act(
    action: np.ndarray,
    state: np.ndarray,
    dt: float,
):
    acc = action[0]  # acceleration
    steer = action[1]  # steering angle

    # Update velocity using acceleration and clip to speed limit
    state[3] += acc * dt
    state[3] = max(min(state[3], SPEED_LIMIT), -SPEED_LIMIT)

    beta = np.arctan(1 / 2 * np.tan(steer))
    velocity = state[3] * np.array([np.cos(state[2] + beta), np.sin(state[2] + beta)])

    state[0:2] += velocity * dt
    state[2] += state[3] * np.sin(beta) / (state[4] / 2) * dt


@njit("f8[:](f8[:])", fastmath=True, cache=True)
def randomise_state(state: np.ndarray):
    state_copy = state.copy()
    state_copy[0] = np.random.uniform(-SCREEN_W / SCALE / 2, SCREEN_W / SCALE / 2)
    state_copy[1] = np.random.uniform(-SCREEN_H / SCALE / 2, SCREEN_H / SCALE / 2)
    state_copy[2] = np.random.uniform(-PI, PI)
    return state_copy


@njit("b1(f8[:,:], f8, f8[:,:,:], f8[:])", fastmath=True, cache=True)
def collision_check(
    ego: np.ndarray, ego_angle: float, others: np.ndarray, others_angle: np.ndarray
):
    # --- AABB filtering (cheap bounding box check) ---
    ego_min_x = ego[:, 0].min()
    ego_max_x = ego[:, 0].max()
    ego_min_y = ego[:, 1].min()
    ego_max_y = ego[:, 1].max()

    for i in range(others.shape[0]):
        other = others[i]
        angle = others_angle[i]

        other_min_x = other[:, 0].min()
        other_max_x = other[:, 0].max()
        other_min_y = other[:, 1].min()
        other_max_y = other[:, 1].max()

        # AABB rejection
        if (
            ego_max_x < other_min_x or ego_min_x > other_max_x or
            ego_max_y < other_min_y or ego_min_y > other_max_y
        ):
            continue

        # --- OBB (Separating Axis Theorem) ---
        def get_rot_mat(theta: float):
            return np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)],
            ])

        ego_rot = get_rot_mat(ego_angle)
        other_rot = get_rot_mat(angle)

        # Use local axes of both rectangles as projection axes
        axes = [
            ego_rot[0],  # Ego's local X
            ego_rot[1],  # Ego's local Y
            other_rot[0],  # Other's local X
            other_rot[1],  # Other's local Y
        ]

        collision = True
        for axis in axes:
            ego_proj = np.dot(ego, axis)
            other_proj = np.dot(other, axis)

            if ego_proj.max() < other_proj.min() or other_proj.max() < ego_proj.min():
                collision = False
                break  # Separating axis found → no collision

        if collision:
            return True

    return False

@njit("f8[:](f8[:], f8)", fastmath=True, cache=True)
def rotate_rad(pos, theta):
    rot_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    return rot_matrix.dot(np.asfortranarray(pos))


@njit("f8[:,:](f8[:])", fastmath=True, cache=True)
def compute_vertices(state: np.ndarray):
    """
    shape of state: (7,)
    """
    l, r, t, b = (
        -state[4] / 2,
        state[4] / 2,
        state[5] / 2,
        -state[5] / 2,
    )
    vertices = np.array([[l, b], [l, t], [r, t], [r, b]])

    for i in range(vertices.shape[0]):
        vertices[i, :] = rotate_rad(vertices[i, :], state[2]) + state[:2]
    return vertices


@njit("f8[:,:](f8[:,:])", fastmath=True, cache=True)
def to_pixel(pos: np.ndarray):
    """
    shape of pos: (num_pos, 2)
    """
    pos_pixel = pos.copy()
    pos_pixel[:, 0] = pos_pixel[:, 0] * SCALE + SCREEN_W / 2
    pos_pixel[:, 1] = pos_pixel[:, 1] * SCALE + SCREEN_H / 2
    return pos_pixel


def draw_rectangle(
    surface: pygame.Surface,
    vertices: np.ndarray,
    color=None,
    obj_type: int = 0,
):
    object_surface = pygame.Surface(surface.get_size(), flags=pygame.SRCALPHA)
    if color is None:
        if obj_type == 0:
            color = YELLOW
        elif obj_type == 1:
            color = BLACK
    pygame.draw.polygon(
        object_surface,
        color,
        vertices,
        width=2,
    )
    gfxdraw.filled_polygon(object_surface, vertices, color)
    
    surface.blit(object_surface, (0, 0))


def draw_direction_pattern(
    surface: pygame.Surface,
    state: np.ndarray,
):
    if state[-1] == 1:
        pass
    else:
        state_copy = state.copy()
        state_copy[0] += np.cos(state[2]) * state[4] / 8 * 3
        state_copy[1] += np.sin(state[2]) * state[4] / 8 * 3
        state_copy[4] = state[4] / 4
        vertices = to_pixel(compute_vertices(state_copy))
        draw_rectangle(surface, vertices, RED)

def draw_wheels(
    surface: pygame.Surface,
    state: np.ndarray,
    wheel_width: float = 0.5,
    wheel_length: float = 1.0,
    color: tuple = BLACK,
):
    """
    Draws 4 wheels slightly protruding outside and placed closer to the center than the car's corners.
    """
    car_x, car_y, car_theta = state[0], state[1], state[2]
    car_length, car_width = state[4], state[5]

    # Closer placement: reduce offset from full length/width
    # Still allow slight protrusion outside car body
    longitudinal_offset = car_length * 0.35  # closer than half length
    lateral_offset = car_width * 0.55        # slightly outside the width

    # Positions relative to car center (before rotation)
    wheel_offsets = np.array([
        [-longitudinal_offset, -lateral_offset],  # rear left
        [-longitudinal_offset,  lateral_offset],  # rear right
        [ longitudinal_offset, -lateral_offset],  # front left
        [ longitudinal_offset,  lateral_offset],  # front right
    ])

    for offset in wheel_offsets:
        wheel_center = rotate_rad(offset, car_theta) + np.array([car_x, car_y])
        l, r = -wheel_length / 2, wheel_length / 2
        b, t = -wheel_width / 2, wheel_width / 2
        wheel_vertices = np.array([[l, b], [l, t], [r, t], [r, b]])

        for i in range(4):
            wheel_vertices[i] = rotate_rad(wheel_vertices[i], car_theta) + wheel_center

        draw_rectangle(surface, to_pixel(wheel_vertices), color)

class ParkingAcc(gym.Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "no_render",
        ],
        "render_fps": FPS,
        "observation_types": [
            "rgb",
            "vector",
        ],
        "action_types": [
            "discrete",
            "multidiscrete",
            "continuous",
            "multicontinuous",
        ],
    }

    def __init__(
        self,
        render_mode: Optional[str] = "no_render",
        observation_type: Optional[str] = "vector",
        action_type: Optional[str] = "discrete",
        bump_on_collision: bool = False,
        add_walls: bool = True,
        obstacles: Optional[np.ndarray] = None,
        rand_start: bool = True,
    ) -> None:
        super().__init__()




        assert render_mode in self.metadata["render_modes"]
        assert observation_type in self.metadata["observation_types"]
        assert action_type in self.metadata["action_types"]
        self.render_mode = render_mode
        self.observation_type = observation_type
        self.action_type = action_type
        self.bump_on_collision = bump_on_collision
        self.rand_start = rand_start

        if type(obstacles) == list:
            obstacles = np.array(obstacles)

        # Compose stationary state from walls and obstacles
        obs = obstacles if obstacles is not None else DEFAULT_OBSTACLES
        if obs.ndim == 1:
            obs = obs[None, :]
        if obs.size == 0:
            if add_walls:
                self.stationary = WALLS.copy()
            else:
                self.stationary = np.empty((0, 7))
        else:
            if add_walls:
                self.stationary = np.vstack([WALLS, obs])
            else:
                self.stationary = obs.copy()

        if observation_type == "vector":
            # Only controlled vehicle: x, y, cos(theta), sin(theta), speed
            self.observation_space = spaces.Box(
                -1.0, 1.0, (5,), dtype=np.float32
            )
        elif observation_type == "rgb":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
            )

        self.action_space = spaces.Discrete(N_SAMPLES_ACC * N_SAMPLES_STEER)

        self.screen = None
        self.surf = None
        self.surf_movable = None
        self.surf_stationary = None
        self.clock = None

    def step(self, action: Union[np.ndarray, int]):
        collision = False
        if action is not None:
            # Prepare the action as before

            acc_action = action // N_SAMPLES_STEER
            steer_action = action % N_SAMPLES_STEER
            action = np.array(
                [ACC_ACTIONS[acc_action], STEERING_ACTIONS[steer_action]]
            )
                
            # Predict next state for collision check
            next_state = self.movable[0].copy()
            
            act(action, next_state, DT)
            next_vertices = compute_vertices(next_state)
            if self.stationary.shape[0] > 0 and collision_check(
                next_vertices,
                next_state[2],
                self.stationary_vertices,
                self.stationary[:, 2],
            ):
                collision = True

            if self.bump_on_collision and collision:
                
                # Ignore the action, set velocity to zero
                self.movable[0, 3]= 0.0
                # Bump backwards
                # self.movable[0, 0] -= np.cos(self.movable[0, 2]) * 0.5
                # self.movable[0, 1] -= np.sin(self.movable[0, 2]) * 0.5
                
                # Do not update position or angle
                # Recompute vertices for rendering/observation
                self.movable_vertices = compute_vertices(self.movable[0])
            else:
                # Normal update: use the already computed next_state
                self.movable[0] = next_state
                self.movable_vertices = next_vertices

        if self.observation_type == "rgb":
            self.obs = self._render("rgb_array", STATE_W, STATE_H)
        elif self.observation_type == "vector":
            obs_vec = np.zeros(5, dtype=np.float32)
            obs_vec[:2] = self.movable[0, :2]
            obs_vec[2] = np.cos(self.movable[0, 2])
            obs_vec[3] = np.sin(self.movable[0, 2])
            obs_vec[4] = self.movable[0, 3]
            obs_vec /= STATE_SCALE[:5]
            self.obs = obs_vec

        reward = self._reward() if not (self.bump_on_collision and collision) else 0.0

        if self.render_mode == "human":
            self.render()
        return self.obs, reward, self.terminated, self.truncated, {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # self.stationary is already set in __init__
        self.stationary_vertices = np.zeros((self.stationary.shape[0], 4, 2))
        for i in range(self.stationary.shape[0]):
            self.stationary_vertices[i] = compute_vertices(self.stationary[i])

        if self.rand_start:
            self.movable = np.array([randomise_state(RAND_INIT_STATE)]) 
        else:
            self.movable = np.array([FIXED_INIT_STATE])
        self.movable_vertices = compute_vertices(self.movable[0])

        if self.stationary.shape[0] > 0:
            while collision_check(
                self.movable_vertices,
                self.movable[0, 2],
                self.stationary_vertices,
                self.stationary[:, 2],
            ):  
                if self.rand_start:
                    self.movable = np.array([randomise_state(RAND_INIT_STATE)])
                else:
                    # Reset to the original position
                    self.movable = np.array([FIXED_INIT_STATE])
                self.movable_vertices = compute_vertices(self.movable[0])
        self.goal_vertices = compute_vertices(GOAL_STATE)

        if self.observation_type == "vector":
            obs_vec = np.zeros(5, dtype=np.float32)
            obs_vec[:2] = self.movable[0, :2]
            obs_vec[2] = np.cos(self.movable[0, 2])
            obs_vec[3] = np.sin(self.movable[0, 2])
            obs_vec[4] = self.movable[0, 3]
            obs_vec /= STATE_SCALE[:5]
            self.obs = obs_vec

        self.terminated = False
        self.truncated = False
        self.run_steps = 0

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def _reward(self):
        reward = 0
        self.run_steps += 1

        if self.stationary.shape[0] > 0:
            if collision_check(
                self.movable_vertices,
                self.movable[0, 2],
                self.stationary_vertices,
                self.stationary[:, 2],
            ):
                self.terminated = True
                reward = -1.0
                return reward
        if collision_check(
            self.movable_vertices,
            self.movable[0, 2],
            np.array([self.goal_vertices]),
            np.array([GOAL_STATE[2]]),
        ):
            self.terminated = True
            reward = 1.0
            return reward
        # if self.run_steps == MAX_STEPS:
        #     self.truncated = True
        #     reward = -1.0
        #     return reward
        return reward

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str, rgb_w: int = SCREEN_W, rgb_h: int = SCREEN_H):

        if not pygame.font.get_init():
            pygame.font.init()
        assert mode in self.metadata["render_modes"]

        if mode == "human" and self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            if self.clock is None:
                self.clock = pygame.time.Clock()

        if mode == "human" or mode == "rgb_array":
            if self.surf_stationary is None:
                self.surf_stationary = pygame.Surface(
                    (SCREEN_W, SCREEN_H), flags=pygame.SRCALPHA
                )
                for i in range(self.stationary.shape[0]):
                    if self.stationary[i, -1] == 0:
                        draw_wheels(
                            self.surf_stationary, self.stationary[i],
                            wheel_width=0.5, wheel_length=0.8, color=BLACK
                        )
                    draw_rectangle(
                        self.surf_stationary,
                        to_pixel(self.stationary_vertices[i]),
                        obj_type=self.stationary[i, -1],
                    )
                    draw_direction_pattern(self.surf_stationary, self.stationary[i])
                # Render the dollar symbol for the goal
                font = pygame.font.SysFont("arial", 32, bold=True)
                text_surf = font.render("$", True, GOLDEN)

                text_surf = pygame.transform.flip(text_surf, False, True)

                # Position it at the goal's center (transformed to pixel coordinates)
                goal_pos_px = GOAL_STATE[:2] * SCALE + np.array([SCREEN_W / 2, SCREEN_H / 2])
                #goal_pos_px[1] = SCREEN_H - goal_pos_px[1]  # Flip Y axis (pygame coordinates)

                # Center the text
                text_rect = text_surf.get_rect(center=goal_pos_px)
                self.surf_stationary.blit(text_surf, text_rect)

            if self.surf_movable is None:
                self.surf_movable = pygame.Surface(
                    (SCREEN_W, SCREEN_H), flags=pygame.SRCALPHA
                )
            self.surf_movable.fill((0, 0, 0, 0))
            draw_wheels(
                self.surf_movable, self.movable[0],
                wheel_width=0.5, wheel_length=0.8, color=BLACK
            )
            draw_rectangle(self.surf_movable, to_pixel(self.movable_vertices), GREEN)
            draw_direction_pattern(self.surf_movable, self.movable[0])

            # NEW: Create final surface and fill background
            surf = pygame.Surface((SCREEN_W, SCREEN_H))
            surf.fill(BACKGROUND_COLOR)
            surf.blit(self.surf_stationary, (0, 0))
            surf.blit(self.surf_movable, (0, 0))
            surf = pygame.transform.flip(surf, False, True)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(BACKGROUND_COLOR)
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(surf, (rgb_w, rgb_h))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        # make the background background color 
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )


if __name__ == "__main__":

    env = ParkingAcc(
        render_mode="human", observation_type="vector", action_type="discrete",
        bump_on_collision=True, rand_start=True
    )
    
    env.reset()
    
    while True:
        action = env.action_space.sample()
        s, r, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break
    env.close()