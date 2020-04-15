#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import random



def create_fns(controlled_agent_id):
    
    def redraw(img):
        if not args.agent_view:
            img = env.render('rgb_array', tile_size=args.tile_size, agent_id=controlled_agent_id)

        window.show_img(img)

    def reset():
        if args.seed != -1:
            env.seed(args.seed)

        obs = env.reset()

        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)
            window.set_caption(env.mission)
        
        if env.multiagent:
            obs = obs[controlled_agent_id]

        redraw(obs)

    def step(action):
        if env.multiagent:
            if type(action) is dict:
                obs, reward, done, info = env.step(action)
            else:    
                # actions = {agent_id: random.randint(0, env.action_space.n - 1) for agent_id in env.agent_ids if not env.agents[agent_id].done}
                actions = {agent_id: 2 if env.env_step_count != 10 else 1 for agent_id in env.agent_ids if not env.agents[agent_id].done}
                actions[controlled_agent_id] = action
                obs, reward, done, info = env.step(actions)
        else:
            obs, reward, done, info = env.step(action)
        reward = reward[controlled_agent_id] if env.multiagent else reward
        print('step=%s, reward=%.2f' % (env.agents[controlled_agent_id].step_count, reward))

        if env.multiagent and (done['__all__'] or done[controlled_agent_id]) or not env.multiagent and done:
            print('done!')
            reset()
        else:
            if env.multiagent:
                obs = obs[controlled_agent_id]
            redraw(obs)

    return redraw, step, reset
def make_key_handler():
    key_a = None
    key_b = None
    def handler(event):
        nonlocal key_a
        nonlocal key_b
        print('pressed', event.key)

        if event.key == 'escape':
            window.close()
            return

        if event.key == 'backspace':
            reset()
            return

        if event.key == 'left':
            key_a = env.actions.left
        if event.key == 'right':
            key_a = env.actions.right
        if event.key == 'up':
            key_a = env.actions.forward

        if event.key == 'a':
            key_b = env.actions.left
        if event.key == 'd':
            key_b = env.actions.right
        if event.key == 'w':
            key_b = env.actions.forward

        if key_a and key_b:
            action = {'agent_1': key_a, 'agent_2':key_b}
            step(action)
            key_a = None
            key_b = None
            return

        # Spacebar
        if event.key == ' ':
            step(env.actions.toggle)
            return
        if event.key == 'pageup':
            step(env.actions.pickup)
            return
        if event.key == 'pagedown':
            step(env.actions.drop)
            return

        if event.key == 'enter':
            step(env.actions.done)
            return
    return handler

def single_agent_key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

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
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)
parser.add_argument(
    '--controlled_agent',
    type=str,
    default='default',
    help="Agent Id to control"
)
parser.add_argument(
    '--multi_control',
    action='store_true',
    help="Agent Id to control"
)

args = parser.parse_args()

env = gym.make(args.env)

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

redraw, step, reset = create_fns(args.controlled_agent)
window = Window('gym_minigrid - ' + args.env)

if args.multi_control:
    key_handler = make_key_handler()
else:
    key_handler = single_agent_key_handler

window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
