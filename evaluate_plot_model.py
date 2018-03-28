import matplotlib.pyplot as plt
import numpy as np

folder = "./data/318/eval_model/"
states_true = np.loadtxt(folder+"states_true.out", delimiter=',')
states_predict = np.loadtxt(folder+"states_predict.out", delimiter=',')

rewards_true = np.loadtxt(folder+"rewards_true.out", delimiter=',')
rewards_predict = np.loadtxt(folder+"rewards_predict.out", delimiter=',')


print("states_true", states_true.shape)
print("states_predict", states_predict.shape)

print("rewards_true", rewards_true.shape)
print("rewards_predict", rewards_predict.shape)

# for i in range(states_true.shape[1]):
#     print("state ", i)
#     idx = i
#     plt.plot(states_true[:, idx])
#     plt.plot(states_predict[:, idx], '--')
#     plt.show()

plt.plot(rewards_true)
plt.plot(rewards_predict, '--')
plt.title("reward")
plt.show()
