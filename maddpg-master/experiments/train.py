import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import pandas as pd
# import tensorflow.nn.rnn_cell as rnn
# import tensorflow.layers as layers
import tensorflow.keras.layers as layers1
import matplotlib.pyplot as plt

import os

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment simple_collaborative_comm simple_spread
    parser.add_argument("--scenario", type=str, default="simple_collaborative_comm", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=150000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.8, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='exp_5', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="policys/policy_fix_fix_2_5_aistudio1/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="policys/policy_fix_fix_2_5_aistudio1/",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment_uav import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        loss = [[] for _ in range(len(trainers))]
        mean_epi_rew = -np.Inf
        best_mean_epi_rew = -np.Inf


        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # print(action_n)
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            # if done or terminal:
            #     obs_n = env.reset()
            #     episode_step = 0
            #     episode_rewards.append(0)
            #     for a in agent_rewards:
            #         a.append(0)
            #     agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode

            for agent in trainers:
                agent.preupdate()
            for i, agent in enumerate(trainers):
                agent_loss = agent.update(trainers, train_step)
                if agent_loss is None:
                    pass
                else:
                    loss[i].append(np.transpose(agent_loss))

                # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                # csv格式保存训练过程数据
                for i, agent in enumerate(trainers):
                    train_traj = pd.DataFrame(loss[i], columns=['q_loss', 'p_loss', 'target_q', 'rew', 'target_q_next',
                                                                'target_q_std'])
                    file_name = arglist.plots_dir + arglist.exp_name + '_agent_' + str(i) + '_traj.csv'
                    # if os.path.exists(file_name):
                    #     train_traj.to_csv(file_name, mode="a", header=False)
                    # else:
                    train_traj.to_csv(file_name, mode='w')


                mean_epi_rew = np.mean(episode_rewards[-arglist.save_rate:])
                if mean_epi_rew > best_mean_epi_rew:
                    best_mean_epi_rew = mean_epi_rew
                    U.save_state(arglist.save_dir, saver=saver, global_step=len(episode_rewards))

                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        round(time.time() - t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))

                print('*************************Bandwidht assignment*****************************')
                for i in env.world.bandwidth:
                    for j in i:
                        print('%10.2e' % j, end='')
                    print('\n', end='')

                print('*************************SNR*****************************')
                for i in env.world.SNR:
                    for j in i:
                        print('%10f' % j, end='')
                    print('\n', end='')
                print('*************************Rate*****************************')
                for i in env.world.Rate:
                    for j in i:
                        print('%10.2e' % j, end='')
                    print('\n', end='')


                # #csv格式保存带宽、SNR、Rate等通信效能数据
                # band_file_name = arglist.plots_dir + arglist.exp_name + '_Bandwidth.csv'
                # SNR_file_name = arglist.plots_dir + arglist.exp_name + '_SNR.csv'
                # Rate_file_name = arglist.plots_dir + arglist.exp_name + '_Rate.csv'
                # band_perform = pd.DataFrame(env.world.bandwidth, columns=['User1', 'User2', 'User3', 'User4', 'User5'])
                # SNR_perfodsrm = pd.DataFrame(env.world.SNR, columns=['User1', 'User2', 'User3', 'User4', 'User5'])
                # Rate_perform = pd.DataFrame(env.world.Rate, columns=['User1', 'User2', 'User3', 'User4', 'User5'])
                # if os.path.exists(file_name):
                #     band_perform.to_csv(band_file_name, mode="a", header=False)
                #     SNR_perform.to_csv(band_file_name, mode="a", header=False)
                #     Rate_perform.to_csv(band_file_name, mode="a", header=False)
                # else:
                #     band_perform.to_csv(band_file_name, mode="w")
                #     SNR_perform.to_csv(band_file_name, mode="w")
                #     Rate_perform.to_csv(band_file_name, mode="w")

                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)

            # saves final episode reward for plotting training curve later
            # if len(episode_rewards) % arglist.save_rate == 0:
            #     rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
            #     with open(rew_file_name, 'wb') as fp:
            #         pickle.dump(final_ep_rewards, fp)
            #     agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
            #     with open(agrew_file_name, 'wb') as fp:
            #         pickle.dump(final_ep_ag_rewards, fp)
            if len(episode_rewards) > arglist.num_episodes:
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
