import numpy as np


def pq(data, P, init_centroids, max_iter):
    N = data.shape[0]
    M = data.shape[1]
    K = init_centroids.shape[1]
    U = int(M/P)
    # print(N, M, K, U)

    codebook = init_centroids
    code = np.empty((N, P))

    # split data by P
    split_data = np.empty((P, N, U))
    for i in range(0, P):
        split_data[i] = data[:, i * U:(i + 1) * U]

        # do k-means on split_data
        last_cluster = np.zeros((N, 1))     # store last centroid for early stop
        x = 0
        while x < max_iter:
            # print(x)
            distance = np.abs((split_data[i] - codebook[i, :, np.newaxis, :])).sum(axis=2)  # l1 distance
            # print((split_data[i] - codebook[i, :, np.newaxis, :]).shape)
            new_cluster = np.argmin(distance, axis=0)                                       # new closed cluster
            # print(new_cluster.shape，new_cluster)

            # early stop
            if (last_cluster == new_cluster).all():
                break
            else:
                last_cluster = (new_cluster[w] for w in range(0, N))

            # update codebook (：new centroid)
            for k in range(0, K):
                if k in new_cluster:
                    codebook[i][k] = (np.median(np.array(split_data[i][new_cluster == k]), axis=0))
                # else:
                    # print("bad cluster: ", k, x,i)
                    # print(codebook[i][k])
            x = x + 1

    for i in range(0, P):
        # update code by k-means
        distance = np.abs(split_data[i] - codebook[i, :, np.newaxis, :]).sum(axis=2)
        code[:, i] = np.argmin(distance, axis=0)
    code = code.astype(np.uint8)
    return codebook, code


def multi_seq(r, s, t, codes):
    visited = {}
    pqueue = []
    out = set()
    i, j = 0, 0
    pqueue.append((r[i][1] + s[j][1], (i, j)))
    # print(pqueue)
    # print([r[i][1] + s[j][1], (r[i][0], s[j][0])])
    # print((r[i][0], s[j][0]), pqueue[(r[i][0], s[j][0])])
    while len(out) < t:
        pqueue = sorted(pqueue, key=lambda x:x[0])
        p = pqueue.pop(0)
        (i, j) = p[1]
        (x, y) = (r[i][0], s[j][0])
        if np.argwhere((codes[:, 0] == x) & (codes[:, 1] == y)).tolist():
            temp = np.argwhere((codes[:, 0] == x) & (codes[:, 1] == y)).tolist()
            for k in temp:
                for x in k:
                    out = out | {x}
                    # print((i, j),(x, y),r[i][1] + s[j][1])
        visited[(i, j)] = True
        # print(visited)
        if i < len(r) - 1 and (j == 0 or visited.get((i+1, j-1))):
            pqueue.append((r[i + 1][1] + s[j][1], (i + 1, j)))
        if j < len(s) - 1 and (i == 0 or visited.get((i-1, j+1))):
            pqueue.append((r[i][1] + s[j + 1][1], (i, j + 1)))
    # print(out)
    return out


def multi_seq_4(queue, t, codes):
    r, s, m, n = queue[0], queue[1], queue[2], queue[3]
    visited = {}
    pqueue = []
    out = set()
    i, j, k, l = 0, 0, 0, 0
    pqueue.append((r[i][1] + s[j][1] + m[k][1] + n[l][1], (i, j, k, l)))
    # print(codes)
    # print(pqueue)
    # print([r[i][1] + s[j][1], (r[i][0], s[j][0])])
    # print((r[i][0], s[j][0]), pqueue[(r[i][0], s[j][0])])
    while len(out) < t:
        pqueue = sorted(pqueue, key=lambda x : x[0])
        p = pqueue.pop(0)
        print(p)
        (i, j, k, l) = p[1]
        print(r, s, m, n)
        (x, y, z, c) = (r[i][0], s[j][0], m[k][0], n[l][0])
        print(x, y, z, c)
        if np.argwhere((codes[:, 0] == x) & (codes[:, 1] == y) & (codes[:, 2] == z) & (codes[:, 3] == c)).tolist():
            temp = np.argwhere((codes[:, 0] == x) & (codes[:, 1] == y) & (codes[:, 2] == z) & (codes[:, 3] == c)).tolist()
            # print(temp)
            for item in temp:
                for elem in item:
                    out = out | {elem}
                    # print((i, j),(x, y),r[i][1] + s[j][1])
        # print(out)
        visited[(i, j)] = True
        # print(visited)
        if i < len(r) - 1 and (j == 0 or visited.get((i+1, j-1, k, l))) \
                and (k == 0 or visited.get((i+1, j, k-1, l))) and (l == 0 or visited.get((i+1, j, k, l-1))):
            pqueue.append((r[i + 1][1] + s[j][1] + m[k][1] + n[l][1], (i + 1, j, k, l)))
        if j < len(s) - 1 and (i == 0 or visited.get((i-1, j+1, k, l)))\
                and (k == 0 or visited.get((i, j+1, k-1,l)))and (l == 0 or visited.get((i, j+1, k, l-1))):
            pqueue.append((r[i][1] + s[j + 1][1] + m[k][1] + n[l][1], (i, j + 1, k, l)))
        if k < len(m) - 1 and (i == 0 or visited.get((i-1, j, k+1, l)))\
                and (j == 0 or visited.get((i, j-1, k+1, l)))and (l == 0 or visited.get((i, j, k+1, l-1))):
            pqueue.append((r[i][1] + s[j][1] + m[k + 1][1] + n[l][1], (i, j, k + 1, l)))
        if l < len(n) - 1 and (i == 0 or visited.get((i-1, j, k, l+1)))\
                and (j == 0 or visited.get((i, j-1, k, l+1)))and (k == 0 or visited.get((i, j, k-1, l+1))):
            pqueue.append((r[i][1] + s[j][1] + m[k][1] + n[l + 1][1], (i, j, k, l + 1)))
    # print(out)
    return out


def query(queries, codebooks, codes, T):
    Q = queries.shape[0]
    M = queries.shape[1]
    P = codes.shape[1]
    U = int(M/P)
    # print(Q, M, P, U)
    split_query = np.empty((Q, P, U))
    candidate = []
    N_set = np.unique(codes, axis=0)
    # print(len(N_set))
    distance = np.zeros((Q, P, len(N_set), 2))  # product quantization
    temp = []
    for x in codes:
        temp.append(tuple(x))
    for i in range(0, Q):
        # queue = np.zeros((P, len(np.unique(codes[:, 0]))))
        queue = []
        for j in range(0, P):
            split_query[i][j] = queries[i, j * U:(j + 1) * U]
            # N_set = np.unique(codes[:, j])
            # print(len(N_set))
            temp_list = []
            # print(len(N_set),np.unique(codes[:, j]))
            for k in range(0, len(N_set)):
                # print(i, j, k)
                distance[i][j][k][0] = N_set[k][j]       # q1 VS u   q2 VS v ......
                # print(np.abs(split_query[i][j] - (codebooks[j][k])).sum())
                distance[i][j][k][1] = np.abs(split_query[i][j] - (codebooks[j][N_set[k][j]])).sum()
            # print(distance[i, j, :, -1].argsort())
            for s in distance[i, j, :, -1].argsort():
                temp_list.append(distance[i][j][s])
                # print(distance[i][j][s][0],distance[i][j][s][1])
            queue.append(temp_list)
            # print(distance[i][j])
        # if P == 2:
        candidate.append(multi_seq(queue[0], queue[1], T, codes))

        # if P == 4:
        #     pass
            # candidate.append(multi_seq_4(queue, T, codes))
    return candidate



import pickle
import time
# # # How to run your implementation for Part 1
with open('./toy_example/Data_File', 'rb') as f:
    Data_File = pickle.load(f, encoding = 'bytes')
with open('./toy_example/Centroids_File', 'rb') as f:
    Centroids_File = pickle.load(f, encoding = 'bytes')
data1 = np.array(Data_File)
centroids = np.array(Centroids_File)
# start = time.time()
codebooks, codes = pq(data1, P=2, init_centroids=centroids, max_iter=20)
# end = time.time()
# time_cost_1 = end - start
#
# print(time_cost_1, codebooks,codes[:10])
# #
# with open('./example/Codebooks_2', 'rb') as f:
#     cdb1 = pickle.load(f, encoding='bytes')
# codebook1 = np.array(cdb1)
# with open('./example/Codes_2', 'rb') as f:
#      cd1= pickle.load(f, encoding='bytes')
# code1 = np.array(cd1)
with open('./toy_example/Query_File', 'rb') as f:
    Query_File = pickle.load(f, encoding = 'bytes')
    # print(type(Query_File),Query_File)
queries = Query_File
# print(type(queries),type(queries[0]))
# start = time.time()
candidates = query(queries, codebooks, codes, T=10)
# end = time.time()
# time_cost_2 = end - start
# print('time', time_cost_2)
# for i in candidates:
print(candidates)
print(len(candidates))
with open('./example/Candidates', 'rb') as f:
     cand2= pickle.load(f, encoding='bytes')
# candidate2 = np.array(cand2)
print("test:",cand2,len(cand2))
