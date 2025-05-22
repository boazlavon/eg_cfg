
## ✅ MBPP Mistake Summary Report

The following table lists tasks where errors were identified in the MBPP benchmark (e.g., incorrect unit tests or flawed reference solutions). Each column shows whether a solution from a given method was correct (`V`) or incorrect (`X`).

### 🧪 Per-Task Evaluation

| Task ID | Sample Solution | UT | BS 1.3b | EG-CFG 1.3b | BS V3 | EG-CFG V3 |
|:--------|:----------------|:--:|:--------:|:-------------:|:------:|:-----------:|
| 180     | X              | X  | V       | V           | V     | V           |
| 276     | V              | X  | V       | V           | V     | V           |
| 310     | X              | X  | V       | V           | V     | V           |
| 312     | V              | X  | V       | V           | V     | V           |
| 313     | X              | X  | V       | V           | V     | V           |
| 436     | X              | X  | V       | V           | V     | V           |
| 438     | X              | X  | X       | V           | X     | V           |
| 461     | V              | X  | V       | V           | V     | V           |
| 493     | V              | X  | X       | X           | X     | V           |

---

### 📊 Aggregate Summary

The table below summarizes how many predictions were correct in each category, and the estimated accuracy improvement (`Δ Accuracy %`) from correcting benchmark issues. These values are based on 500 MBPP samples total.

#### ✅ Total Correct Per Method

| Method        | Sample Solution | Unit Tests | Baseline 1.3B | EG-CFG 1.3B | Baseline V3 | EG-CFG V3 |
|---------------|------------------|-------------|----------------|--------------|--------------|-------------|
| ✅ Correct     | –                | –           | 7              | 8            | 7            | 9           |

#### 📈 Accuracy Delta (Δ Accuracy %) — out of 500 samples

| Method        | Baseline 1.3B | EG-CFG 1.3B | Baseline V3 | EG-CFG V3 |
|---------------|----------------|--------------|--------------|-------------|
| Δ Accuracy %  | +1.4%          | +1.6%        | +1.4%        | +1.8%       |
