def count_bidirectional(lst):
    count = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i][0] == lst[j][1] and lst[i][1] == lst[j][0]:
                count += 1
    return str(count)