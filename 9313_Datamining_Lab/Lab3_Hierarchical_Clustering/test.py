## import modules here
import numpy as np
################# Question 1 #################


def dot_product(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res


def hc(data, k):# do not change the heading of the function
    row = len(data)
    similarity = []
    for i in range(0, row):
        similarity.append([])
        for j in range(0, row):
            if i == j:
                similarity[i].append(0)
            else:
                similarity[i].append(dot_product(data[i], data[j]))
    similarity = np.array(similarity)
    cluster = []
    for i in range(0, row):
        cluster.append([i])

    while(len(cluster) != k):
        new_matrix = []                 # update new similarity matrix
        for i in range(0, len(cluster)):
            new_matrix.append([])
            for j in range(0, len(cluster)):
                if i >= j:
                    new_matrix[i].append(0.0)
                else:
                    if len(cluster[i]) > 1 or len(cluster[j]) > 1:       # indexs that already clustered, caculate new value
                        sub = []
                        for p in cluster[i]:
                            sub.append(p)
                        for p in cluster[j]:
                            sub.append(p)
                        # print("sub is ",sub)
                        sim = 0
                        for m in range(0, len(sub)-1):
                            for n in range(m+1, len(sub)):
                                if sub[m] > sub[n]:
                                    sim = sim+similarity[sub[n]][sub[m]]
                                else:
                                    sim = sim + similarity[sub[m]][sub[n]]
                        sim = (sim*2)/(len(sub)*(len(sub)-1))
                        new_matrix[i].append(sim)
                    else:
                        new_matrix[i].append(similarity[cluster[i][0]][cluster[j][0]])
        [x, y] = np.where(new_matrix == np.max(new_matrix))
        for line in new_matrix:
            print(line)

        row_index = x[0]
        column_index = y[0]
        cluster[row_index] = cluster[row_index] + cluster[column_index]
        del cluster[column_index]
        print(cluster)
    cout = 0
    result = []
    for i in range(0,len(data)):
        result.append(0)

    for i in range(0,len(cluster)):
        for j in range(0,len(cluster[i])):
            result[cluster[i][j]] = cout
        cout = cout + 1
    return result


data = np.loadtxt('asset/data_3.txt', dtype=float)
# for i in range(1, 8):
print(hc(data, 2))