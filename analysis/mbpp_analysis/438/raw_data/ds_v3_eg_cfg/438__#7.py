def count_bidirectional(sequence):
    count = 0
    seen = set()
    for pair in sequence:
        reversed_pair = (pair[1], pair[0])
        if reversed_pair in seen:
            count += 1
            seen.remove(reversed_pair)
        else:
            seen.add(pair)
    return str(count)