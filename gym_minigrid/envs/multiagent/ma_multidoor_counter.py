from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from ray.rllib.env import MultiAgentEnv


class MultidoorCounter(MiniGridEnv, MultiAgentEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, height=4, width=8):
        super().__init__(
            height=height,
            width=width,
            max_steps=10*height*width,
            see_through_walls=True,
            multiagent=True,
            agent_ids=['agent_1', 'agent_2']
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(agent_id='agent_1', color='red'), width - 2, 1)
        self.put_obj(Goal(agent_id='agent_2', color='green'), width - 2, height - 2)

        # # Create a vertical splitting wall
        # splitIdx = self._rand_int(2, width-2)
        self.grid.horz_wall(1, 2, length=width-2, obj_type=Counter)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.agents['agent_1'].pos = (1, 1)
        self.agents['agent_1'].dir = 3

        self.agents['agent_2'].pos = (1, 3)
        self.agents['agent_2'].dir = 3

        # Place a door in the wall
        self.put_obj(Door('yellow', is_locked=True), 3, 1)
        self.put_obj(Door('blue', is_locked=True), 5, 1)
        self.put_obj(Door('green', is_locked=True), 7, 1)

        # Place a door in the wall
        self.put_obj(Key('yellow'), 3, 3)
        self.put_obj(Key('blue'), 5, 3)
        self.put_obj(Key('green'), 7, 3)

        self.mission = "use the keys to open the doors and then get to the goal"


class MultidoorCounter8x5(MultidoorCounter):
    def __init__(self):
        super().__init__(height=5, width=10)


register(
    id='MiniGrid-MA-MultidoorCounter-8x8-v0',
    entry_point='gym_minigrid.envs:MultidoorCounter8x5'
)
