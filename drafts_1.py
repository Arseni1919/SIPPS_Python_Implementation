import numpy as np
import random
import time

N = 1000
l_np = np.random.random((N, N))

l_list = []
for i in range(N):
    l_list.append([])
    for j in range(N):
        l_list[i].append(random.random())


S = 1000

start_time = time.time()
for i in range(S):
    x, y = random.randint(0, N-1), random.randint(0, N-1)
    item = l_np[x, y]
    if item in l_np:
        print(f'', end='')
diff_np = time.time() - start_time
print(f'\nt np: {diff_np: .2f}')
start_time = time.time()
for i in range(S):
    x, y = random.randint(0, N-1), random.randint(0, N-1)
    item = l_list[x][y]
    for tat_l in l_list:
        if item in tat_l:
            print(f'', end='')
            break
diff_list = time.time() - start_time
print(f'\nt list: {diff_list: .2f}')
print(f'{diff_list/diff_np=: .2f}')





# start_time = time.time()
# for i in range(S):
#     x, y = random.randint(0, N-1), random.randint(0, N-1)
#     print(f'\r{l_np[x, y]}', end='')
# print(f'\nt np: {time.time() - start_time}')
# start_time = time.time()
# for i in range(S):
#     x, y = random.randint(0, N-1), random.randint(0, N-1)
#     print(f'\r{l_list[x][y]}', end='')
# print(f'\nt list: {time.time() - start_time}')