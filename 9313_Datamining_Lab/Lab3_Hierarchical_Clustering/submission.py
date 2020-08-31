## import modules here
import numpy as np


################# Question 1 #################


def dot_product(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res


def isInter(a,b):
    result = list(set(a) & set(b))
    if result:
        return True
    else:
        return False


def exist(x,list):
    for i in list:
        if x in i:
            return True
    return False


def hc(data, k):  # do not change the heading of the function
    row = len(data)
    similarity = []
    for i in range(0, row):
        similarity.append([])
        for j in range(0, row):
            if i == j:
                similarity[i].append(0)
            else:
                similarity[i].append(dot_product(data[i], data[j]))

    result = []
    if k == 1:
        for i in range(0, len(data)):
            result.append(0)
        return result

    cluster = []
    # [x, y] = np.where(similarity == np.max(similarity))
    # # for line in new_matrix:
    # #     print(line)
    # row_index = x[0]
    # column_index = y[0]
    # cluster.append([row_index, column_index])

    while (len(cluster) != row-k):
        [x, y] = np.where(similarity == np.max(similarity))
        # for line in new_matrix:
        #     print(line)
        row_index = x[0]
        column_index = y[0]
        cluster.append([row_index, column_index])
        # print(len(cluster))
        for i in range(0, row):
            for j in range(0, row):
                if i >= j:
                    similarity[i][j] = 0
                else:
                    if i == column_index:
                        similarity[i][j] = 0
                    if j == column_index:
                        similarity[i][j] = 0
                    if i == row_index:
                        p = max(similarity[j][row_index], similarity[row_index][j])
                        q = max(similarity[j][column_index], similarity[column_index][j])
                        similarity[i][j] = min(p, q)
                    if j == row_index:
                        p = max(similarity[i][row_index], similarity[row_index][i])
                        q = max(similarity[i][column_index], similarity[column_index][i])
                        similarity[i][j] = min(p, q)

        # for line in similarity:
        #     print(line)

    # print(cluster)
    count = 1
    while(count != 0):
        count = 0
        for i in range(0,len(cluster)-1):
            index = []
            for j in range(i+1, len(cluster)):
                if isInter(cluster[i], cluster[j]):
                    count = count + 1
                    index.append(i)
                    index.append(j)
                    cluster[i] = list(set(cluster[i]+cluster[j]))
                    del cluster[j]
                    break
            if count != 0:
                break
    for i in range(0, row):
        if not exist(i, cluster):
            cluster.append([i])
    # print(cluster)
    cout = 0
    result = []
    for i in range(0, len(data)):
        result.append(0)

    for i in range(0, len(cluster)):
        for j in range(0, len(cluster[i])):
            result[cluster[i][j]] = cout
        cout = cout + 1
    # print(cluster)
    # print(result)
    return result

#
# data = np.loadtxt('asset/data_2.txt', dtype=float)
# for i in range(1, 18):
#     print(hc(data, i))