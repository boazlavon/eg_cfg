# MBPP Task 436 — Print Negative Numbers in a List

## 🧾 Original Prompt

**Instruction:**
```
Write a python function to print negative numbers in a list.
```

**Sample Solution:**
```python
def neg_nos(list1):
  for num in list1: 
    if num < 0: 
       return num
```

**Unit Tests:**
```python
assert neg_nos([-1,4,5,-6]) == -1,-6
assert neg_nos([-1,-2,3,4]) == -1,-2
assert neg_nos([-7,-6,8,9]) == -7,-6
```

---

## ❌ Issues in the Original Task

- **Reference Solution:** Returns only the **first** negative number due to an early `return` and does not **print** anything, contradicting the prompt【156†source】.
- **Unit Tests:** Use invalid syntax like `== -1,-6`, which Python interprets incorrectly as a tuple comparison. Tests also imply the function should return multiple values directly, which is not how Python behaves【157†source】.

---

## 🤖 Model Outputs and Justification

### DeepSeek-Coder 1.3B (Baseline, γ = 0.0)
```python
def neg_nos(lst):
    for i in lst:
        if i < 0:
            print(i)
```
**Test Outcome:** 0/3 tests passed【158†source】  
**Explanation:** The function correctly prints each negative number, as per the prompt, but fails due to flawed test assertions expecting tuple-style returns.

---

### DeepSeek-V3-0324 (Baseline, γ = 0.0)
```python
def neg_nos(lst):
    negative_numbers = []
    for num in lst:
        if num < 0:
            negative_numbers.append(num)
    for neg_num in negative_numbers:
        print(neg_num, end=", ")
    print()
```
**Test Outcome:** 0/3 tests passed【159†source】  
**Explanation:** This version prints all negative numbers cleanly and correctly. It fails only because the benchmark tests are syntactically and logically incorrect.

---

## ✅ Final Judgment

**Correct Model(s):**  
✔️ DeepSeek-Coder 1.3B (Baseline)  
✔️ DeepSeek-V3-0324 (Baseline)

**Summary:**  
Both model implementations fulfill the task’s intent: they identify and print all negative numbers in the input list. All benchmark test failures are due to invalid unit test expectations, not errors in the logic of the code.
