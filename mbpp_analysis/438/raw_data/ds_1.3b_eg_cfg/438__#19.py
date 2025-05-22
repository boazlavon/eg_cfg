def count_bidirectional(tuples):
    count = 0
    for i in range(len(tuples)):
        for j in range(i + 1, len(tuples)):
            if tuples[i][0] == tuples[j][1] and tuples[i][1] == tuples[j][0]:
                count += 1
    return count