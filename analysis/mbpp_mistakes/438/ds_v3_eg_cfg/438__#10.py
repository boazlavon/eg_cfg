def count_bidirectional(li):
    count = 0
    seen = set()
    for pair in li:
        reversed_pair = (pair[1], pair[0])
        if reversed_pair in seen:
            count += 1
            seen.remove(reversed_pair)
        else:
            seen.add(tuple(pair))
    return str(count * 2)