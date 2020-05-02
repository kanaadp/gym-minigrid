from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class MACounterCirc(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        max_steps=200
    ):
        super().__init__(
            width=6,
            height=5,
            max_steps=max_steps,
            see_through_walls=True,
            multiagent=True,
            agent_ids=['agent_1', 'agent_2']
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        self.grid.horz_wall(2, 2, length=2, obj_type=Counter)

        self.grid.horz_wall(2, 0, length=2, obj_type=Counter)
        self.grid.horz_wall(2, 4, length=2, obj_type=Counter)

        agent_placement = self._rand_bool()

        self.agents[self.agent_ids[int(agent_placement)]].pos = (1, 2)
        self.agents[self.agent_ids[1 - int(agent_placement)]].pos = (4, 2)
        self.agents[self.agent_ids[int(agent_placement)]].dir = 0
        self.agents[self.agent_ids[1 - int(agent_placement)]].dir = 2

        self.mission = "deliver the boxes and balls"

        self.grid.get(2, 0).place(Ball(color='blue'))
        self.grid.get(3, 4).place(Box(color='red'))
        self.box_landed = False
        self.ball_landed = False

    def step(self, agent_actions):
        obs_dict, reward_dict, done_dict, info_dict = super().step(agent_actions)

        # movin balls:
        if not self.grid.get(2, 0).has_obj() and self.ball_landed:
            self.grid.get(2, 0).place(Ball(color='blue'))
            self.ball_landed = False

        if self.grid.get(2, 4).has_obj() and self.grid.get(2, 4).obj.type == 'ball':
            self.grid.get(2, 4).retrieve()
            self.ball_landed = True
            if info_dict['agent_1']['action_info'] == ('drop_counter', 'ball'):
                reward_dict['agent_1'] += 1
            elif info_dict['agent_2']['action_info'] == ('drop_counter', 'ball'):
                reward_dict['agent_2'] += 1

        # movin boxes:
        if not self.grid.get(3, 4).has_obj() and self.box_landed:
            self.grid.get(3, 4).place(Box(color='red'))
            self.box_landed = False
        if self.grid.get(3, 0).has_obj() and self.grid.get(3, 0).obj.type == 'box':
            self.grid.get(3, 0).retrieve()
            self.box_landed = True
            if info_dict['agent_1']['action_info'] == ('drop_counter', 'box'):
                reward_dict['agent_1'] += 1
            elif info_dict['agent_2']['action_info'] == ('drop_counter', 'box'):
                reward_dict['agent_2'] += 1

        return obs_dict, reward_dict, done_dict, info_dict


    def dense_reward_fn(self, rewards):
        return {agent_id: 0 for agent_id in rewards}

register(
    id='MiniGrid-MA-MACounterCirc-v0',
    entry_point='gym_minigrid.envs.multiagent.ma_countercirc:MACounterCirc'
)