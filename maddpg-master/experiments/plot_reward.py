import pickle
import matplotlib.pyplot as plt
import seaborn as sns

f = open('learning_curves/exp_23/rewards.pkl', 'rb')
# f1 = open('./learning_curves/exp_4_1/rewards.pkl','rb')
# f2 = open('./learning_curves/rewards.pkl','rb')

rewards = pickle.load(f)
# rewards1 = pickle.load(f1)
print(rewards[-1])
# rew = rewards+rewards1
plt.plot(rewards)

plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('Convergence of Rewards')
# plt.legend(['Aistudio', 'fix'])
plt.show()
