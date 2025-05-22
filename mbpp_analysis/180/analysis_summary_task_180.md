# MBPP Task 180 — Distance Between Two Coordinates

## 🧾 Original Prompt

**Instruction:**
```
Write a function to calculate distance between two points using latitude and longitude.
```

**Sample Solution:**
```python
from math import radians, sin, cos, acos
def distance_lat_long(slat, slon, elat, elon):
    dist = 6371.01 * acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon))
    return dist
```

**Unit Tests:**
```python
assert distance_lat_long(23.5,67.5,25.5,69.5)==12179.372041317429
assert distance_lat_long(10.5,20.5,30.5,40.5)==6069.397933300514
assert distance_lat_long(10,20,30,40)==6783.751974994595
```

---

## ❌ Issues in the Original Task

- **Unit Tests:** All three expected values are incorrect for both the Haversine formula and the spherical law of cosines, due to not converting degrees to radians.
- **Sample Solution:** Uses trigonometric functions on degree inputs, which leads to incorrect results unless the inputs are preconverted to radians.

---

## 🤖 Model Outputs and Justification

### DeepSeek-Coder 1.3B (Baseline, γ = 0.0)
```python
import math

def distance_lat_long(lat1, long1, lat2, long2):
    # Convert degrees to radians
    lat1, long1, lat2, long2 = map(math.radians, [lat1, long1, lat2, long2])

    # Haversine formula
    dlat = lat2 - lat1
    dlong = long2 - long1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlong/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = 6371 * c
    return distance
```
**Test Outcome:** All tests failed, not due to incorrect code but due to flawed unit tests.  
**Explanation:** This is a clean and correct implementation of the Haversine formula. It properly converts degrees to radians and uses the standard Earth radius. The failure is purely due to incorrect benchmark expectations.

---

### DeepSeek-V3-0324 (Baseline, γ = 0.0)
```python
import math

def distance_lat_long(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    R = 6371.0
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance
```
**Test Outcome:** All tests failed  
**Explanation:** This is an accurate Haversine implementation and handles input in degrees correctly. Like with 1.3B, it is marked incorrect due to benchmark errors, not logic errors.

---

## ✅ Final Judgment

**Correct Model(s):**  
✔️ DeepSeek-Coder 1.3B (Baseline)  
✔️ DeepSeek-V3-0324 (Baseline)

**Summary:**  
Both models produce correct solutions using the Haversine formula. The failures are caused by incorrect expected values in the MBPP test cases, which assume a different or incorrect distance formula. These outputs should be marked as correct in a properly validated benchmark.
