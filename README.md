# Occam

**"Shave away the complexity."**

**Occam** is a ruthless complexity scanner designed to cut through the noise 
and reveal the true performance of your Python algorithms. It combines **
Dynamic Runtime Profiling** (measuring growth rates against hostile patterns) 
with **Static Bytecode Analysis** to tell you not just *how fast* your code is, 
but *why* it behaves that way.

## Installation

Occam requires Python 3.9+ and standard visualization libraries.

```bash
# Clone the repository
git clone [https://github.com/antonhibl/occam.git](https://github.com/antonhibl/occam.git)
cd occam

# Install dependencies
pip install matplotlib numpy scipy
```

## Quick Start

Run Occam on a single Python script. By default, it generates random integer 
inputs.

```bash
python3 scanner.py my_script.py --max_n 10000
```

**The Output:**

1. A raw performance table (Time, Memory, Growth Factors).
2. A generated `battle_random_report.png` visualizing the exact curve of 
   your complexity.

## Battle Mode (Compare Solutions)

Pit multiple algorithms against each other. See exactly where one solution 
crushes the other one.

```bash
python3 scanner.py solution_v1.py solution_v2.py --type list --max_n 50000 --output comparison.png
```

## Torture Testing

Your code works on random data, but does it break under pressure? Occam 
generates hostile patterns to expose weak points (like Quicksort's 
worst case).

```bash
# Test strictly sorted inputs (often the worst case for partition logic)
python3 scanner.py solution.py --type list --pattern sorted

# Test reverse-sorted inputs (exposes Timsort cheats vs manual loops)
python3 scanner.py solution.py --type list --pattern reverse
```

## Static Analysis (X-Ray)

Use the `--dump` flag to see what the Python interpreter is actually doing. 
This reveals the "Interpreter Tax"—why a "clean" Python one-liner might be 
slower than a verbose loop.

```bash
python3 scanner.py solution.py --dump
```

**Metrics Revealed:**

* **Cyclomatic Complexity:** Measures the number of linearly independent paths 
                         through the code. A higher score indicates higher 
                         structural complexity and increased minimum 
                         testing requirements.
* **Max Loop Nesting:** The maximum depth of nested iteration structures (AST 
                    nodes). This serves as a static proxy for lower-bound time 
                    complexity (e.g., depth 2 implies at least $O(N^2)$ unless 
                    the loop range is constant).
* **Bytecode Instructions:** The raw count of CPython opcodes the virtual 
                         machine must execute.

## How to Prepare Your Scripts

To analyze a specific function, wrap your solution in a class with a 
`run_analysis_entry` method. This allows you to preprocess data (e.g., 
splitting a list into two) *before* the timer starts.

```python
from typing import List

class Solution:
    # 1. Occam calls this method first
    def run_analysis_entry(self, data: List[int]) -> float:
        # Perform setup (not counted in algorithmic complexity logic)
        mid = len(data) // 2
        nums1 = sorted(data[:mid])
        nums2 = sorted(data[mid:])
        
        # 2. Call the algorithm you actually want to test
        return self.findMedianSortedArrays(nums1, nums2)

    def findMedianSortedArrays(self, nums1, nums2):
        # ... your logic here ...

```

## Reading the Charts

* **Time (Left Graph):**
* **Flat Line:** O(Log(N)) or O(1).
* **Straight Diagonal:** O(N) or Polynomial.
* **J-Curve:** Exponential or Factorial.
* **Memory (Right Graph):**
* **Sawtooth Pattern:** Unstable memory usage, usually caused by **Recursive 
                    Slicing** (creating copies of lists, then freeing them).
* **Flat Line:** O(1) space.
