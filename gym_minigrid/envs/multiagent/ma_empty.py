from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class MAEmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=16,
        agent_start_poses=[(1,1), (8, 8)],
        agent_start_dirs=[0, 2],
        num_agents=2
    ):
        self.agent_start_poses = agent_start_poses
        self.agent_start_dirs = agent_start_dirs

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True,
            multiagent=True,
            agent_ids=['agent_1', 'agent_2']
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(agent_id='agent_1'), width - 2, height - 2)

        self.put_obj(Goal(agent_id='agent_2'), 1, 1)

        for i, agent_id in enumerate(self.agent_ids):
            # Place the agent
            if self.agent_start_poses is not None:
                self.agents[agent_id].pos = self.agent_start_poses[i]
                self.agents[agent_id].dir = self.agent_start_dirs[i]
            else:
                self.place_agent(agent_id)

        self.mission = "get to the green goal square"

register(
    id='MiniGrid-MA-Empty-8x8-v0',
    entry_point='gym_minigrid.envs.multiagent.ma_empty:MAEmptyEnv'
)