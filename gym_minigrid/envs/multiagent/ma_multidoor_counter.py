from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from ray.rllib.env import MultiAgentEnv


class MultidoorCounter(MiniGridEnv, MultiAgentEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, height=4, width=8, randomize_key_pos=False):
        self.randomize_key_pos = randomize_key_pos
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

        # # Create a horizontal splitting wall
        self.grid.horz_wall(1, 2, length=width-2, obj_type=Counter)

        self.agents['agent_1'].pos = (1, 1)
        self.agents['agent_1'].dir = 3

        self.agents['agent_2'].pos = (1, 3)
        self.agents['agent_2'].dir = 3

        self.put_obj(Door('yellow', is_locked=True), 3, 1)
        self.put_obj(Door('blue', is_locked=True), 5, 1)
        self.put_obj(Door('green', is_locked=True), 7, 1)

        if self.randomize_key_pos:
            key_pos = np.random.choice(np.arange(2, 8), 3, replace=False)
        else:
            key_pos = [3, 5, 7]

        self.put_obj(Key('yellow'), key_pos[0], height - 2)
        self.put_obj(Key('blue'), key_pos[1], height - 2)
        self.put_obj(Key('green'), key_pos[2], height - 2)
    

        self.mission = "use the keys to open the doors and then get to the goal"

    @classmethod
    def calc_dense_reward(cls, info_dict):
        dense_rewards = {}
        if 'agent_1' in info_dict:
            action_info = info_dict['agent_1']['action_info']
            if action_info[0] == 'pickup_counter' and action_info[1].type == 'key':
                dense_rewards['agent_1'] = 0.01
            elif action_info[0] == 'door' and action_info[1]:
                dense_rewards['agent_1'] = 0.05
            elif action_info[0] == 'door':
                dense_rewards['agent_1'] = 0.01
            else:
                dense_rewards['agent_1'] = 0.00

        if 'agent_2' in info_dict:
            action_info = info_dict['agent_2']['action_info']
            if action_info[0] == 'pickup' and action_info[1].type == 'key':
                dense_rewards['agent_2'] = 0.01
            elif action_info[0] == 'drop_counter' and action_info[1].type == 'key':
                dense_rewards['agent_2'] = 0.01
            else:
                dense_rewards['agent_2'] = 0.00

        return dense_rewards

class MultidoorCounter5x11(MultidoorCounter):
    def __init__(self):
        super().__init__(height=5, width=11)

class MultidoorCounter6x11(MultidoorCounter):
    def __init__(self):
        super().__init__(height=6, width=11)

class MultidoorCounter6x11Random(MultidoorCounter):
    def __init__(self):
        super().__init__(height=6, width=11, randomize_key_pos=True)


register(
    id='MiniGrid-MA-MultidoorCounter-5x11-v0',
    entry_point='gym_minigrid.envs:MultidoorCounter5x11'
)

register(
    id='MiniGrid-MA-MultidoorCounter-6x11-v0',
    entry_point='gym_minigrid.envs:MultidoorCounter6x11'
)

register(
    id='MiniGrid-MA-MultidoorCounter-Random-6x11-v0',
    entry_point='gym_minigrid.envs:MultidoorCounter6x11Random'
)
