
import gymnasium as gym

def register_all():

    """
    Register all created custom environments that haven't been registered yet.
    """
      
gym.register(
    id="MiniGrid-12x12-v0",
    entry_point="environments.minigrid.mini_grid:ObstaclesGridEnv",
    max_episode_steps=1000000000,
    kwargs=dict(
        size=12,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        bump_penalty=0,
        obstacles=[(3,3), (3,4), (3,5)],
    ),
)


gym.register(
    id="CustomFrozenLakeNoHoles4x4-v1",
    entry_point="environments.frozenlake.frozen_lake:CustomFrozenLakeEnv",
    kwargs={
        "desc": [
            "SFFF",
            "FFFF",
            "FFFF",
            "FFFG"
            ],
        "map_name": None,
        "is_slippery": False,
        "terminate_on_hole": False,
        "hole_reward": 0,
    },
)



gym.register(
    id="CustomFrozenLakeNoHoles8x8-v1",
    entry_point="environments.frozenlake.frozen_lake:CustomFrozenLakeEnv",
    kwargs={
        "desc": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFG"
            ],
        "map_name": None,
        "is_slippery": False,
        "terminate_on_hole": False,
        "hole_reward": 0,
    },
)

gym.register( # 16x16 empty grid
id="CustomFrozenLakeNoHoles16x16-v1",
entry_point="environments.frozenlake.frozen_lake:CustomFrozenLakeEnv",
kwargs={
    "desc": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
        ],
    "map_name": None,
    "is_slippery": False,
    "terminate_on_hole": False,
    "hole_reward": 0,
},
)

gym.register( # 20x20 empty grid
    id="CustomFrozenLakeNoHoles20x20-v1",
    entry_point="environments.frozenlake.frozen_lake:CustomFrozenLakeEnv",
    kwargs={
        "desc": [
            "SFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFFFFFFFG"
            ],
            
        "map_name": None,
        "is_slippery": False,
        "terminate_on_hole": False,
        "hole_reward": 0,
    },
)

gym.register(
    id="DefaultFrozenLake4x4-v1",
    entry_point="environments.frozenlake.frozen_lake:CustomFrozenLakeEnv",
    kwargs={
        "map_name": "4x4",
        "is_slippery": False,
        "hole_reward": 0,
        "terminate_on_hole": False,
    },
)

gym.register(
    id="CustomLunarLander",
    entry_point="environments.lunarlander.lunar_lander:CustomLunarLander",
    kwargs={
        "gravity": -10.0,
        "enable_wind": False,
        "wind_power": 15.0,
        "turbulence_power": 1.5,
        "num_asteroids": 0,
        "ignore_obstacle_collisions": False,
        "penalize_obstacle_collisions": False,
    },
)

gym.register(
    id="ParkingEnv",
    entry_point="environments.highwayenv.parking:ParkingEnv",
)

gym.register(
    id="ParkingSimple",
    entry_point="environments.highwayenv.parking_new:ParkingSimple",
)

gym.register(
    id="ParkingAcc",
    entry_point="environments.highwayenv.parking_acc:ParkingAcc",
)

#  self,
#         render_mode: Optional[str] = None,
#         continuous: bool = False,
#         gravity: float = -10.0,
#         enable_wind: bool = False,
#         wind_power: float = 15.0,
#         turbulence_power: float = 1.5,
#         num_asteroids: int = 0,
#         ignore_obstacle_collisions: bool = False,
#         penalize_obstacle_collisions: bool = False,  # New flag

if "DefaultFrozenLake8x8-v1" not in gym.registry:
    gym.register(
        id="DefaultFrozenLake8x8-v1",
        entry_point="environments.frozenlake.frozen_lake:CustomFrozenLakeEnv",
        kwargs={
            "map_name": "8x8",
            "is_slippery": False,
            "hole_reward": 0,
            "terminate_on_hole": False,
        },
    )
