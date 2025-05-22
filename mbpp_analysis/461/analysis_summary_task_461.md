# MBPP Task 461 — Count Uppercase Characters in a String

## 🧾 Original Prompt

**Instruction:**
```
Write a python function to count the upper case characters in a given string.
```

**Sample Solution:**
```python
def upper_ctr(str):
    upper_ctr = 0
    for i in range(len(str)):
        if str[i] >= 'A' and str[i] <= 'Z':
            upper_ctr += 1
        return upper_ctr
```

**Unit Tests:**
```python
assert upper_ctr('PYthon') == 1
assert upper_ctr('BigData') == 1
assert upper_ctr('program') == 0
```

---

## ❌ Issues in the Original Task

- **Reference Solution:** The `return` is incorrectly indented inside the loop, so the function exits after checking only the first character. This causes it to undercount uppercase characters【170†source】.
- **Unit Tests:** Expected counts are incorrect. `'PYthon'` contains 2 uppercase characters (`'P'`, `'Y'`), and `'BigData'` contains 2 (`'B'`, `'D'`). The test expectations claim only 1 in each, which is incorrect【170†source】.

---

## 🤖 Model Outputs and Justification

### DeepSeek-Coder 1.3B (Baseline, γ = 0.0)
```python
def upper_ctr(s):
    return sum(1 for c in s if c.isupper())
```
**Test Outcome:** 1/3 passed【172†source】  
**Explanation:** This is a clean and correct implementation using `.isupper()`. It fails two tests because the expected values in the benchmark are incorrect.

---

### DeepSeek-V3-0324 (Baseline, γ = 0.0)
```python
def upper_ctr(s):
    count = 0
    for char in s:
        if char.isupper():
            count += 1
    return count
```
**Test Outcome:** 1/3 passed【173†source】  
**Explanation:** Also correct, using a standard loop with `.isupper()`. Again, the test failures are due to faulty benchmark expectations.

---

## ✅ Final Judgment

**Correct Model(s):**  
✔️ DeepSeek-Coder 1.3B (Baseline)  
✔️ DeepSeek-V3-0324 (Baseline)

**Summary:**  
Both model outputs correctly count uppercase characters. The reference solution has a logic bug, and the benchmark tests expect incorrect values. The model completions are valid and should be accepted.
