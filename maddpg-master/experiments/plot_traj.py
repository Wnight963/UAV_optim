import pickle
import matplotlib.pyplot as plt
import pandas as pd

f = open('./learning_curves/agent_0_traj.csv','rb')
data = pd.read_csv('./learning_curves/agent_0_traj.csv')
# print(data)

# plt.plot(data.index,data.q_loss,data.index,data.p_loss)
plt.plot(data.target_q_std)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('reward')
plt.title('Rewards')
plt.legend(['Aistudio'])
plt.show()
