#!/usr/bin/env python3

from threading import Thread, Lock
import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import random


class TwoPlayerEnvController(Thread):
    def __init__(self, dx, window, agent_names=['agent_1', 'agent_2']):
        assert len(agent_names) == 2, "Two Players Only!"
        self.agent_names = agent_names
        self.dx = dx
        self.curr_action_lock = Lock()
        self.action_list = []
        self.window = window
        self.done = {agent_name: False for agent_name in self.agent_names}
        self.reset_curr_action()

    def reset_curr_action(self):
        self.curr_action = {self.agent_names[0]: MiniGridEnv.Actions.no_op,
                            self.agent_names[1]: MiniGridEnv.Actions.no_op}

    def update_action(self, agent_id, action):
        self.curr_action_lock.acquire()
        self.curr_action[agent_id] = action
        self.curr_action_lock.release()

    def run(self):
        while not self.window.closed:
            self.curr_action_lock.acquire()
            curr_action = {agent_id: self.curr_action[agent_id]
                           for agent_id in self.curr_action if not self.done[agent_id]}

            self.reset_curr_action()
            self.curr_action_lock.release()
            self.action_list.append(curr_action)
            self.step(curr_action)

            time.sleep(self.dx)

    def redraw(self):
        img = env.grid.render(
            args.tile_size,
            [env.agents[agent_id].pos for agent_id in env.agent_ids],
            [env.agents[agent_id].dir for agent_id in env.agent_ids],
            ['red', 'green']
        )
        self.window.show_img(img)

    def reset(self):
        if args.seed != -1:
            env.seed(args.seed)

        env.reset()
        self.done = {agent_name: False for agent_name in self.agent_names}

        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)
            window.set_caption(env.mission)

        self.redraw()

    def step(self, action):
        assert type(action) is dict
        obs, reward_dict, self.done, info = env.step(action)

        if self.done['__all__']:
            print('done!')
            self.reset()
        else:
            self.redraw()

    def get_key_handler(self):
        def key_handler(event):

            if event.key == 'escape':
                self.window.close()
                return

            if event.key == 'backspace':
                self.reset()
                return

            if event.key == 'enter':
                self.step(env.actions.done)
                return

            if event.key == 'left':
                key_a = env.actions.left
            elif event.key == 'right':
                key_a = env.actions.right
            elif event.key == 'up':
                key_a = env.actions.forward
            elif event.key == ' ':
                key_a = env.actions.toggle
            elif event.key == '/':
                key_a = env.actions.pickup
            elif event.key == '.':
                key_a = env.actions.drop
            else:
                key_a = None

            if event.key == 'a':
                key_b = env.actions.left
            elif event.key == 'd':
                key_b = env.actions.right
            elif event.key == 'w':
                key_b = env.actions.forward
            elif event.key == 'x':
                key_b = env.actions.toggle
            elif event.key == 'e':
                key_b = env.actions.pickup
            elif event.key == 'c':
                key_b = env.actions.drop
            else:
                key_b = None

            if key_a is not None:
                self.update_action(self.agent_names[0], key_a)
            elif key_b is not None:
                self.update_action(self.agent_names[1], key_b)

        return key_handler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        help="size at which to render tiles",
        default=32
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    env = gym.make(args.env)

    window = Window('gym_minigrid - ' + args.env)

    controller = TwoPlayerEnvController(dx=0.1, window=window)

    key_handler = controller.get_key_handler()

    window.reg_key_handler(key_handler)

    controller.reset()

    controller.run()

    # Blocking event loop
    window.show(block=True)
