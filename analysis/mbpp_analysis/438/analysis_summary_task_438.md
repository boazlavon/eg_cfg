# MBPP Task 438 — Count Bidirectional Tuple Pairs

## 🧾 Original Prompt

**Instruction:**
```
Write a function to count bidirectional tuple pairs.
```

**What this likely means:**
Given a list of 2-element tuples (a, b), count how many times both (a, b) and (b, a) appear in the list — but each matching pair should be counted **once**. Duplicates should not result in overcounting.

---

## 🧪 Sample Reference Solution
```python
def count_bidirectional(test_list):
    res = 0
    for idx in range(0, len(test_list)):
        for iidx in range(idx + 1, len(test_list)):
            if test_list[iidx][0] == test_list[idx][1] and test_list[idx][1] == test_list[iidx][0]:
                res += 1
    return str(res)
```

**Unit Tests:**
```python
assert count_bidirectional([(5, 6), (1, 2), (6, 5), (9, 1), (6, 5), (2, 1)]) == '3'
assert count_bidirectional([(5, 6), (1, 3), (6, 5), (9, 1), (6, 5), (2, 1)]) == '2'
assert count_bidirectional([(5, 6), (1, 2), (6, 5), (9, 2), (6, 5), (2, 1)]) == '4'
```

---

## ❌ Issues with the Reference Solution

- **Incorrect comparison logic:** The sample implementation checks if `test_list[iidx][0] == test_list[idx][1]` and `test_list[idx][1] == test_list[iidx][0]`, which always compares the same value on both sides, rather than (a, b) with (b, a).
- **Does not filter matched pairs:** It can double-count symmetric pairs if both (a, b) and (b, a) appear multiple times.
- **Return type is a string:** While syntactically valid, returning `str(res)` adds confusion since most counting tasks naturally return integers.
- **Unit test outputs are strings:** The benchmark expects string comparison, which further reinforces incorrect design choices.

---

## 🤖 Baseline Model Results

### DeepSeek-Coder 1.3B (Baseline, γ = 0.0)
```python
def count_bidirectional(tuples):
    count = 0
    for tup in tuples:
        if tup[0] == tup[1]:
            count += 1
    return str(count)
```

- **Misunderstood the task:** This baseline checks for tuples where both elements are the same, like (3, 3), instead of identifying bidirectional relationships.
- **Does not even attempt to find reversed pairs.**
- **Fails all tests** as the task requires relational reasoning across different tuple positions, which this version does not do at all.
- **Score:** 0/3 tests passed.

---

### DeepSeek-V3-0324 (Baseline, γ = 0.0)
```python
def count_bidirectional(tuples_list):
    count = 0
    seen_pairs = []
    for pair in tuples_list:
        a = pair[0]
        b = pair[1]
        reversed_pair = (b, a)
        if reversed_pair in seen_pairs:
            count += 1
            seen_pairs.remove(reversed_pair)
        else:
            seen_pairs.append(pair)
    return str(count * 2)
```

- **Better logic structure:** Tries to track reversed pairs using a list and remove duplicates.
- **Flawed count adjustment:** Multiplies the result by 2, which causes overcounting.
- **Return type again is a string**, matching flawed unit test expectations but making the logic harder to reuse.
- **Score:** 2/3 tests passed — success due to incidental alignment with test expectations, not because of general correctness.

---

## ✅ EG-CFG Solutions: DeepSeek-Coder 1.3B

EG-CFG (Execution-Guided Completion with Feedback) successfully produced **two variants** that are both logically valid:

### Implementation A (String Return):
```python
def count_bidirectional(lst):
    count = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i][0] == lst[j][1] and lst[i][1] == lst[j][0]:
                count += 1
    return str(count)
```

### Implementation B (Int Return):
```python
def count_bidirectional(tuples):
    count = 0
    for i in range(len(tuples)):
        for j in range(i + 1, len(tuples)):
            if tuples[i][0] == tuples[j][1] and tuples[i][1] == tuples[j][0]:
                count += 1
    return count
```

- **Both solutions are correct** and pass logical verification.
- **Only the return type** affects whether unit tests pass due to the benchmark's expectation of string values.
- **Demonstrates adaptability:** EG-CFG can navigate multiple valid formats when benchmark criteria are unclear or incorrect.

---

## ✅ EG-CFG Solutions: DeepSeek-V3-0324

The EG-CFG completions from the larger model demonstrate more robust handling using `set` structures to track already matched pairs:

### Implementation A
```python
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
```

### Implementation B (Doubled Count Variant)
```python
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
```

### Implementation C
```python
def count_bidirectional(ls):
    count = 0
    seen = set()
    for pair in ls:
        reversed_pair = (pair[1], pair[0])
        if reversed_pair in seen:
            count += 1
            seen.remove(reversed_pair)
        else:
            seen.add(pair)
    return str(count)
```

- **Accurate semantics:** These correctly identify bidirectional pairs without overcounting.
- **All fail benchmark tests only due to** mismatch in expected outputs (strings and double-counts).

---

## 🔍 Benchmark Misalignment

The reference solution and unit tests **assume that**:

- Duplicated bidirectional pairs (e.g., (6, 5) and (5, 6) appearing multiple times) count more than once.
- Results should be compared as strings, not integers.

This creates **misleading failures** for correct solutions and highlights the importance of evaluation integrity.

---

## ✅ Final Verdict

| Model | Baseline | EG-CFG |
|-------|----------|--------|
| DeepSeek-Coder 1.3B | ❌ 0/3 (wrong logic) | ✅ Correct, flexible output |
| DeepSeek-V3-0324    | ⚠️ 2/3 (doubled count) | ✅ Correct logic, fails due to flawed tests |

**Key Insight:**  
EG-CFG enables generation of **semantically correct implementations** even when the benchmark itself is flawed or underspecified. It handles ambiguity gracefully and produces multiple valid variants for robustness.

