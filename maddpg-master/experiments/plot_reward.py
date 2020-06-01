import pickle
import matplotlib.pyplot as plt

f = open('./learning_curves/experiments_rewards.pkl','rb')
f1 = open('./learning_curves/experiments_agrewards.pkl','rb')
rewards = pickle.load(f)
agrewards = pickle.load(f1)
plt.plot(rewards)
plt.plot(agrewards)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('reward')
plt.title('Rewards')
plt.show()
