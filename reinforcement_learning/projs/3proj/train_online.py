
import time
import cv2
import pdb
import gym

from recoder import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
from utils.train import set_seed_everywhere
from utils.environment import get_agent_types
from overcooked_ai_py.env import OverCookedEnv

# from overcooked_ai_py.env import OverCookedEnv
# from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
# from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
# from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair
# from overcooked_ai_py.agents.benchmarking import AgentEvaluator
# from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

## Swap between the 3 layouts here:
# layout = "cramped_room"
# layout = "asymmetric_advantages"
# layout = "forced_coordination"

# Length of Episodes.  Do not modify for your submission!
# Modification will result in a grading penalty!
horizon = 400

reward_shaping = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 1,
    "POT_DISTANCE_REW": 1,
    "SOUP_DISTANCE_REW": 1,
    "DID_SOMETHING_REW": 0.1,
}

# Build the environment.  Do not modify!
# mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
# base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
# env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

from model.utils.model import *

from utils.agent import find_index

import hydra
from omegaconf import DictConfig

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'Workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.discrete_action = cfg.discrete_action_space
        self.save_replay_buffer = cfg.save_replay_buffer
        # self.env = NormalizedEnv(make_env(cfg.env, discrete_action=self.discrete_action))
        self.env = OverCookedEnv(scenario=self.cfg.env, episode_length=horizon)
        self.env_agent_types = get_agent_types(self.env)
        self.agent_indexes = find_index(self.env_agent_types, 'ally')
        self.adversary_indexes = find_index(self.env_agent_types, 'adversary')

        # OU Noise settings
        self.num_seed_steps = cfg.num_seed_steps
        self.ou_exploration_steps = cfg.ou_exploration_steps
        self.ou_init_scale = cfg.ou_init_scale
        self.ou_final_scale = cfg.ou_final_scale

        if self.discrete_action:
            cfg.agent.params.obs_dim = self.env.observation_space.n
            # cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
            cfg.agent.params.action_dim = self.env.action_space.n
            cfg.agent.params.action_range = list(range(cfg.agent.params.action_dim))
        else:
            # Don't use!
            cfg.agent.params.obs_dim = self.env.observation_space[0].shape[0]
            cfg.agent.params.action_dim = self.env.action_space[0].shape[0]
            cfg.agent.params.action_range = [-1, 1]

        cfg.agent.params.agent_index = self.agent_indexes
        cfg.agent.params.critic.input_dim = cfg.agent.params.obs_dim + cfg.agent.params.action_dim

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.common_reward = cfg.common_reward
        obs_shape = [len(self.env_agent_types), cfg.agent.params.obs_dim]
        action_shape = [len(self.env_agent_types), cfg.agent.params.action_dim if not self.discrete_action else 1]
        reward_shape = [len(self.env_agent_types), 1]
        dones_shape = [len(self.env_agent_types), 1]
        self.replay_buffer = ReplayBuffer(obs_shape=obs_shape,
                                          action_shape=action_shape,
                                          reward_shape=reward_shape,
                                          dones_shape=dones_shape,
                                          capacity=int(cfg.replay_buffer_capacity),
                                          device=self.device)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0


    def evaluate(self):
        average_episode_reward = 0

        self.video_recorder.init(enabled=True)
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            episode_step = 0

            done = False
            episode_reward = 0
            while not done:
                action = self.agent.act(obs, sample=True)
                obs, rewards, done, info = self.env.step(action)
                rewards = np.array(info['shaped_r_by_agent']).reshape(-1, 1)

                self.video_recorder.record(self.env)

                episode_reward += sum(rewards)[0]
                episode_step += 1

            average_episode_reward += episode_reward
        self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps + 1:
            if done or self.step % self.cfg.eval_frequency == 0:
                print("steps %d, episode %d, episode_reward %.2f" % (self.step, episode, episode_reward))
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    start_time = time.time()

                # pdb.set_trace()
                if self.step != 0:
                    if info:
                        soups = sum([len(x) for x in info['episode']['ep_game_stats']['soup_delivery']])
                    else:
                        soups = 0
                else:
                    soups = 0
                if soups > 0:
                    print("We made a soup!")
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/soups', soups, self.step)

                obs = self.env.reset()
                self.ou_percentage = max(0, self.ou_exploration_steps - (self.step - self.num_seed_steps)) / self.ou_exploration_steps
                print(self.ou_percentage)
                self.agent.scale_noise(self.ou_final_scale + (self.ou_init_scale - self.ou_final_scale) * self.ou_percentage)
                self.agent.reset_noise()

                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            if self.step < self.cfg.num_seed_steps:
                action = np.array([self.env.action_space.sample() for _ in self.env_agent_types])
                if self.discrete_action: action = action.reshape(-1, 1)
            else:
                agent_observation = obs[self.agent_indexes]
                agent_actions = self.agent.act(agent_observation, sample=True)
                action = agent_actions

            if self.step >= self.cfg.num_seed_steps and self.step >= self.agent.batch_size:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, rewards, done, info = self.env.step(action)
            rewards = np.array(info['shaped_r_by_agent']).reshape(-1, 1)

            if episode_step + 1 == self.env.episode_length:
                done = True

            if self.cfg.render:
                cv2.imshow('Overcooked', self.env.render())
                cv2.waitKey(1)

            episode_reward += sum(rewards)[0]

            if self.discrete_action: action = action.reshape(-1, 1)

            dones = np.array([done for _ in self.env.agents]).reshape(-1, 1)

            self.replay_buffer.add(obs, action, rewards, next_obs, dones)

            obs = next_obs
            episode_step += 1
            self.step += 1

            if self.step % 5e4 == 0 and self.save_replay_buffer:
                self.replay_buffer.save(self.work_dir, self.step - 1)
        
        # one last run to save the final video
        self.video_recorder.init(enabled=True)
        obs = self.env.reset()
        episode_step = 0
        done = False
        while not done:
            action = self.agent.act(obs, sample=False)
            obs, rewards, done, info = self.env.step(action)
            rewards = np.array(info['shaped_r_by_agent']).reshape(-1, 1)
            self.video_recorder.record(self.env)
            episode_step += 1
        print("here")
        self.video_recorder.save(f'finalvideo_{self.step}.mp4')


@hydra.main(config_path='config', config_name='train')
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
