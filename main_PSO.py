from PSO import PSO_core
import numpy as np
import matplotlib.pyplot as plt
import multiagent.scenarios as scenarios
from multiagent.PSO_env_uav import MultiAgentEnv

def process_action(act_raw):
    act=[]
    act_n = np.array_split(act_raw, 3)
    for item in act_n:
        act.append([softmax(item[0:5]), softmax(item[5:])])
    return act

def softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)

scenario_name = "PSO_collaborative_comm"
scenario = scenarios.load(scenario_name + ".py").Scenario()
world = scenario.make_world()
env = MultiAgentEnv(world,reward_callback=scenario.reward)

pop_size = 30
act_dim = 45
max_iter = 300
x_bound = [0, 1]
v_bound = [0, 1]
PSO_brain = PSO_core(pop_size, act_dim, max_iter, x_bound, v_bound)
action_pops = PSO_brain.init_pop()
fitness = []
for action in action_pops:
    act = process_action(action)
    fitness.append(env.step(act))
PSO_brain.init_best(fitness)

iter_count = 1
record_g_fitness_best = []

while iter_count < max_iter:
    action_pops = PSO_brain.optimize()
    fitness = []
    for action in action_pops:
        act = process_action(action)
        fitness.append(env.step(act))
    # print(fitness)
    PSO_brain.feed_fitness(fitness)
    g_best, g_fitness_best = PSO_brain.update()
    record_g_fitness_best.append(g_fitness_best)
    iter_count += 1
    # if iter_count % 5 == 0:
    #     PSO_brain.w = PSO_brain.w*PSO_brain.w_dec
    #     PSO_brain.c1 = PSO_brain.c1 * PSO_brain.w_dec
    #     PSO_brain.c2 = PSO_brain.c2 * PSO_brain.w_dec
    if iter_count % 1 == 0:
        # print('gbest: ',PSO_brain.x_pop)
        # print(fitness)
        print('\n*************************Bandwidht assignment*****************************')
        for i in env.world.bandwidth:
            for j in i:
                print('%10.2e' % j, end='')
            print('\n', end='')

        print('*************************SNR*****************************')
        for i in env.world.SNR:
            for j in i:
                print('%10.2f' % j, end='')
            print('\n', end='')
        print('*************************Rate*****************************')
        for i in env.world.Rate:
            for j in i:
                print('%10.2e' % j, end='')
            print('\n', end='')
        print('iter: ', iter_count, 'finished!')


# print(record_g_fitness_best)
# print(g_best)
# g_best, g_fitness_best, record_g_fitness_best = PSO_brain.evolve()
# print('g_best: ', g_best)
# print('fitness: ', g_fitness_best)
plt.figure(1)
plt.plot(record_g_fitness_best)
plt.grid()
plt.show()