def quick_sort_test(queue,codes,t):
    P = len(queue)
    pqueue = []
    r = queue[0]
    s = queue[1]
    for j in range(0, len(r)):
        for k in range(0, len(s)):
            temp = [(r[j][1] + s[k][1]), [r[j][0].tolist(), s[k][0].tolist()]]
            pqueue.append(temp)
    # print(pqueue)
    pqueue = sorted(pqueue, key=lambda x: x[0])
    # print(len(pqueue))
    out = set()

    while len(out) < t:
        p = pqueue.pop(0)
        # print('p', p)
        for i in range(0, len(codes)):
            if tuple(p[1]) == codes[i]:
                out = out | {i}
                # print(out)
    return out


def quick_sort(queue,codes,t):
    P = len(queue)
    # print(P)
    pqueue = [[]]
    visited = {}
    i = 0
    for j in range(0,len(queue[i])):
        temp = (queue[i][j][1], [queue[i][j][0].tolist()])
        pqueue[i].append(temp)
    print('queue:',pqueue)
    # return 0
    while i < P-1:
        r = pqueue[i]
        s = queue[i+1]
        pqueue.append([])
        for j in range(0,len(r)):
            for k in range(0, len(s)):
                temp = ((r[j][0] + s[k][1]),r[j][1]+[s[k][0].tolist()])
                pqueue[i+1].append(temp)

        i += 1
    pqueue[-1] = sorted(pqueue[-1], key=lambda x: x[0])
    # pqueue[-1].sort(key=takefirst)
    out = set()
    while len(out) < t:
        p = pqueue[-1].pop(0)
        # print(p)
        code_index = ()
        for x in p[1]:
            code_index = code_index + (x,)
        for x in range(0,len(codes)):
            if tuple(codes[x]) == code_index:
                # print(x,code_index)
                out = out | {x}
    return out