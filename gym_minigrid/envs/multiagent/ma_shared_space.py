from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from ray.rllib.env import MultiAgentEnv


class SharedSpace(MiniGridEnv, MultiAgentEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, height=4, width=8, max_steps=30, randomize_key_pos=False):
        self.randomize_key_pos = randomize_key_pos
        super().__init__(
            height=height,
            width=width,
            max_steps=max_steps,
            see_through_walls=True,
            multiagent=True,
            agent_ids=['agent_1', 'agent_2']
        )
        self.scaling = 1e-3

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.wall_rect(2, 1, 1, 1)
        self.grid.wall_rect(width - 3, 1, 1, 1)
        self.grid.horz_wall(2, 2, length=width-4)
        self.grid.horz_wall(0, height - 1, obj_type=Counter)
        self.grid.vert_wall(0, 0, obj_type=Counter)
        self.grid.vert_wall(width - 1, 0, obj_type=Counter)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(agent_id='agent_1', color='red'), width - 2, 1)
        self.put_obj(Goal(agent_id='agent_2', color='green'), 1, 1)

        # # Create a horizontal splitting wall
        # self.grid.horz_wall(1, 2, length=width-2, obj_type=Counter)

        self.agents['agent_1'].pos = (1, height - 2)
        self.agents['agent_1'].dir = 0

        self.agents['agent_2'].pos = (width - 2, height - 2)
        self.agents['agent_2'].dir = 0

        self.put_obj(Door('yellow', is_locked=True), 1, 2)
        self.put_obj(Door('blue', is_locked=True), 5, 2)
        # # self.put_obj(Door('green', is_locked=True), 7, 1)

        if self.randomize_key_pos:
            key_pos = np.random.choice(np.arange(1, width - 2), 2, replace=False)
        else:
            key_pos = [2, 4]

        self.grid.get(key_pos[0], height - 1).place(Key("yellow"))
        self.grid.get(key_pos[1], height - 1).place(Key("blue"))

        self.mission = "use the keys to open the doors and then get to the goal"

    def dense_reward_fn(self, info_dict):
        dense_rewards = {}
        if 'agent_1' in info_dict:
            action_info = info_dict['agent_1']['action_info']
            if action_info[0] == 'pickup_counter' and action_info[1].type == 'key':
                dense_rewards['agent_1'] = 1
            elif action_info[0] == 'door' and action_info[1]:
                dense_rewards['agent_1'] = 5
            elif action_info[0] == 'door':
                dense_rewards['agent_1'] = 1
            else:
                dense_rewards['agent_1'] = 0
            dense_rewards['agent_1'] *= self.scaling

        if 'agent_2' in info_dict:
            action_info = info_dict['agent_2']['action_info']
            if action_info[0] == 'pickup_counter' and action_info[1].type == 'key':
                dense_rewards['agent_2'] = 1
            elif action_info[0] == 'door' and action_info[1]:
                dense_rewards['agent_2'] = 5
            elif action_info[0] == 'door':
                dense_rewards['agent_2'] = 1
            else:
                dense_rewards['agent_2'] = 0
            dense_rewards['agent_2'] *= self.scaling

        return dense_rewards


class SharedSpace7x7(SharedSpace):
    def __init__(self):
        super().__init__(height=7, width=7, max_steps=100)


class SharedSpace6x8(SharedSpace):
    def __init__(self):
        super().__init__(height=6, width=11, max_steps=100)


class SharedSpace6x11Random(SharedSpace):
    def __init__(self):
        super().__init__(height=6, width=11, max_steps=30, randomize_key_pos=True)


register(
    id='MiniGrid-MA-SharedSpace-7x7-v0',
    entry_point='gym_minigrid.envs:SharedSpace7x7',
    reward_threshold=None
)

register(
    id='MiniGrid-MA-SharedSpace-6x11-v0',
    entry_point='gym_minigrid.envs:SharedSpace6x11',
    reward_threshold=None
)

register(
    id='MiniGrid-MA-SharedSpace-Random-6x11-v0',
    entry_point='gym_minigrid.envs:SharedSpace6x11Random',
    reward_threshold=None
)
