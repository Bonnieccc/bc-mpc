import pickle
from data_buffer import DataBuffer

filename = './hahah.txt'

########### Saving ############
# data_buffer = DataBuffer()

# for i in range(10):
#     data_buffer.add(1, 2, 5)
# print("buffer size: ", data_buffer.size)

# file = open('./hahah.txt', 'w') 
# pickle.dump(data_buffer, file) 



########### Loading ############

file = open(filename, 'r') 
hahah = pickle.load(file) 
print(hahah.size)