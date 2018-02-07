import numpy as np
from data_buffer import DataBuffer, DataBuffer_general 

data_buffer = DataBuffer_general(5 ,2)

for i in range(10):
    data_buffer.add((i, i+1))
    print(data_buffer.size)
    print(data_buffer.sample(10))
