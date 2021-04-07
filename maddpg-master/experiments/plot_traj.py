import pickle
import matplotlib.pyplot as plt
import pandas as pd

f = open('./learning_curves/exp_17/agent_1_traj.csv','rb')
data = pd.read_csv('./learning_curves/exp_17/agent_1_traj.csv')
# print(data)

# plt.plot(data.index,data.q_loss,data.index,data.p_loss)
plt.plot(data.p_loss)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('reward')
plt.title('Rewards')
plt.legend(['Aistudio'])
plt.show()
