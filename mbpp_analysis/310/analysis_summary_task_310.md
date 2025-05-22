# MBPP Task 310 — Convert a String to a Tuple

## 🧾 Original Prompt

**Instruction:**
```
Write a function to convert a given string to a tuple.
```

**Sample Solution:**
```python
def string_to_tuple(str1):
    result = tuple(x for x in str1 if not x.isspace()) 
    return result
```

**Unit Tests:**
```python
assert string_to_tuple("python 3.0") == ('p', 'y', 't', 'h', 'o', 'n', '3', '.', '0')
assert string_to_tuple("item1") == ('i', 't', 'e', 'm', '1')
assert string_to_tuple("15.10") == ('1', '5', '.', '1', '0')
```

---

## ❌ Issues in the Original Task

- **Reference Solution:** Incorrectly filters out whitespace characters. The prompt says to convert the string to a tuple, which should include all characters—including spaces【112†source】.
- **Unit Tests:** The expected output assumes spaces are removed, which contradicts the standard interpretation of the prompt. Therefore, both the reference implementation and the test expectations are flawed【113†source】.

---

## 🤖 Model Outputs and Justification

### DeepSeek-Coder 1.3B (Baseline, γ = 0.0)
```python
def string_to_tuple(s):
    return tuple(s)
```
**Test Outcome:** 2/3 tests passed【114†source】  
**Explanation:** The implementation is correct and matches the natural reading of the prompt. It fails one test because the test incorrectly expects whitespace to be removed.

---

### DeepSeek-V3-0324 (Baseline, γ = 0.0)
```python
def string_to_tuple(input_string):
    return tuple(input_string)
```
**Test Outcome:** 2/3 tests passed【115†source】  
**Explanation:** Also correct. It preserves all characters including whitespace as expected by the prompt. Test failure is due to incorrect benchmark expectations.

---

## ✅ Final Judgment

**Correct Model(s):**  
✔️ DeepSeek-Coder 1.3B (Baseline)  
✔️ DeepSeek-V3-0324 (Baseline)

**Summary:**  
Both model outputs correctly interpret the prompt by preserving all characters in the input string. The benchmark incorrectly penalizes correct implementations due to faulty test expectations that assume whitespace should be removed.

