# --------------------------------------------------------
#     程序：粒子群算法（PSO）解决多无人机位置部署优化与通信资源分配问题
#     作者：王玮健
#     日期：2021.3.31
#     语言：Python 3.6
# --------------------------------------------------------
import numpy as np


class PSO_core(object):
    def __init__(self, pop_size, act_dim, max_iter, x_bound, v_bound):
        self.w = 0.2  # 惯性权重
        self.c1 = 0.6  # 个体学习因子
        self.c2 = 0.2  # 群体学习因子
        self.w_dec = 0.99

        self.pop_size = pop_size  # 种群规模
        self.dim = act_dim  # 动作维度
        self.max_iter = max_iter  # 搜索次数
        self.x_bound = x_bound  # 动作取值范围
        self.v_bound = v_bound  # 速度取值范围

        self.x_pop = None
        self.v_pop = None
        self.fitness = None
        self.p_best = None
        self.g_best = None
        self.p_fitness_best = None
        self.g_fitness_best = None
        self.record_g_best = []
        self.record_g_fitness_best = []

    def init_pop(self):
        self.x_pop = np.random.uniform(self.x_bound[0], self.x_bound[1],(self.pop_size, self.dim))  # 初始化个体位置
        self.v_pop = np.random.rand(self.pop_size, self.dim)  # 初始化个体速度
        return self.x_pop

    def init_best(self, fitness):
        self.feed_fitness(fitness)
        self.p_best = self.x_pop  # 个体最佳位置
        # print(self.fitness)
        # print(np.argmax(self.fitness))
        self.g_best = self.x_pop[np.argmax(self.fitness)]  # 群体最佳位置
        self.p_fitness_best = self.fitness  #个体最佳适应度
        self.g_fitness_best = self.fitness[np.argmax(self.fitness)]  # 群体最佳适应度


    def calcu_fitness(self):
        fitness = []
        # for i in self.x_pop:
        #     fitness.append(-np.linalg.norm(i-np.array([0.57, 0.34, 0.2642, 0.8489, 0.215,-0.548,-0.874]), 2))
        return np.array(fitness)

    def feed_fitness(self,fitness):
        self.fitness = np.array(fitness)

    def optimize(self):
        r1 = np.random.rand(self.pop_size, self.dim)  # 个体影响随机因子
        r2 = np.random.rand(self.pop_size, self.dim)  # 群体影响随机因子
        # 更新速度和位置
        self.v_pop = self.w * self.v_pop + self.c1 * r1 * (self.p_best - self.x_pop) + self.c2 * r2 * (
                    self.g_best - self.x_pop)
        self.v_pop = np.clip(self.v_pop, self.v_bound[0], 0.2*self.v_bound[1])

        self.x_pop = self.x_pop + self.v_pop
        self.x_pop = np.clip(self.x_pop, self.x_bound[0], self.x_bound[1])
        return self.x_pop

    def update(self):
        # 更新p_best和g_best,记录最优fitness
        index = np.greater(self.fitness, self.p_fitness_best)
        # print(index)
        self.p_fitness_best[index] = self.fitness[index]
        self.p_best[index] = self.x_pop[index]

        if self.fitness[np.argmax(self.fitness)] > self.g_fitness_best:
            self.g_fitness_best = self.fitness[np.argmax(self.fitness)]
            self.g_best = self.x_pop[np.argmax(self.fitness)]
        return self.g_best, self.g_fitness_best



    def evolve(self):

        iter_count = 1

        while(iter_count < self.max_iter):

            r1 = np.random.rand(self.pop_size, self.dim)  # 个体影响随机因子
            r2 = np.random.rand(self.pop_size, self.dim)  # 群体影响随机因子


            # 更新速度和位置
            self.v_pop = self.w * self.v_pop + self.c1*r1*(self.p_best-self.x_pop) + self.c2*r2*(self.g_best-self.x_pop)
            self.v_pop = np.clip(self.v_pop, self.v_bound[0], self.v_bound[1])

            self.x_pop = self.x_pop + self.v_pop
            self.x_pop = np.clip(self.x_pop, self.x_bound[0], self.x_bound[1])

            self.fitness = self.calcu_fitness()

            index = np.greater(self.fitness, self.p_fitness_best)
            self.p_fitness_best[index] = self.fitness[index]
            self.p_best[index] = self.x_pop[index]

            if self.fitness[np.argmax(self.fitness)] > self.g_fitness_best:
                self.g_fitness_best = self.fitness[np.argmax(self.fitness)]
                self.g_best = self.x_pop[np.argmax(self.fitness)]

            self.record_g_fitness_best.append(self.g_fitness_best)
            iter_count +=1
            print('iter: ', iter_count, 'finished!')

        return self.g_best, self.g_fitness_best, self.record_g_fitness_best


