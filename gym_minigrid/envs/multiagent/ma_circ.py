from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class MACircEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        num_agents=2,
        max_steps=40
    ):
        super().__init__(
            grid_size=size,
            max_steps=max_steps,
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
        
        for i in range(2, height-2): 
            self.grid.horz_wall(2, i, length=width-4)

        # Place a goal square in the bottom-right corner
        self.place_obj(Goal(agent_id='agent_1', color='red'))

        self.place_obj(Goal(agent_id='agent_2', color='green'))

        for i, agent_id in enumerate(self.agent_ids):
            self.place_agent(agent_id=agent_id)       

        self.mission = "get to the green goal square"

    def dense_reward_fn(self, rewards):
        return {agent_id: 0 for agent_id in rewards}

register(
    id='MiniGrid-MA-Circ-8x8-v0',
    entry_point='gym_minigrid.envs.multiagent.ma_circ:MACircEnv'
)