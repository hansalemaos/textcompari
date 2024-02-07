# A function to get the difference between two files and return a pandas DataFrame

## pip install textcompari

### Tested against Windows 10 / Python 3.11 / Anaconda 

### Important!
The module will be compiled when you import it for the first time. Cython and a C/C++ compiler must be installed!

```python
from rapidfuzz import fuzz
from textcompari import get_file_diff

"""
A function to get the difference between two files and return a pandas DataFrame.

:param afile: A file to compare (str, bytes, tuple, list, np.ndarray)
:param bfile: Another file to compare (str, bytes, tuple, list, np.ndarray)
:param window_shifts: The number of shifts for the window (default 5)
:param min_fuzz_match: The minimum fuzzy match score (default 80)
:param fuzz_scorer: The fuzzy scorer function (default fuzz.WRatio)
:param cpus: The number of CPUs to use (default 5)
:return: A pandas DataFrame containing the difference between the files
"""


afile = r"C:\Users\hansc\Downloads\difffindertest\test1_1.txt"
bfile = r"C:\Users\hansc\Downloads\difffindertest\test1_2.txt"

df = get_file_diff(
    afile, bfile, window_shifts=300, min_fuzz_match=99, fuzz_scorer=fuzz.WRatio, cpus=5
)
print(df)


```