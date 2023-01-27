# Simple Calculator of Similarity

---

## About The Project
> 파이썬 Dictionary 기반의 유사도 계산 모듈

### Calculator
   - `Jaccard`
   - `Cosine`
   - `Euclidean`

## Built with
- macOS Monterey 12.4 (M1)
- Python >= 3.9

## Library Installation
- `pip install scipy scikit-learn tqdm`

## Example
```python
from similarity import SimilarityCalculator

s = SimilarityCalculator()

doc = {"a": [1, 2], "b": [2]}
s.calculate(method="jaccard", data=doc)
>>> {'a': {'b': 0.5}, 'b': {'a': 0.5}}

doc = {"a": {"x": 1, "y": 3}, "b": {"x": 2}}
s.calculate(method="euclidean", data=doc)
>>> {'a': {'b': 3.1622776601683795}, 'b': {'a': 3.1622776601683795}}

doc = {"a": [1, 2], "b": [2, 2]}
s.calculate(method="cosine", data=doc)
>>> {'a': {'b': 0.9486832980505137}, 'b': {'a': 0.9486832980505137}}
```