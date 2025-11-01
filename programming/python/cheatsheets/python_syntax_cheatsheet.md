# Python Syntax Cheatsheet - Professional Reference

## Variables & Data Types

```python
# Variable assignment
name = "Alice"
age = 30
price = 19.99
is_active = True
empty = None

# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 0

# Type hints (Python 3.5+)
name: str = "Alice"
count: int = 10
prices: list[float] = [9.99, 19.99]
```

## String Operations

```python
# String formatting (modern)
name, age = "Alice", 30
message = f"{name} is {age} years old"
message = f"{price:.2f}"  # 2 decimal places
message = f"{name:>10}"   # Right align, width 10

# Common methods
text.lower()
text.upper()
text.strip()              # Remove whitespace
text.split(",")           # Split into list
", ".join(items)          # Join list into string
text.replace("old", "new")
text.startswith("prefix")
text.endswith("suffix")
"sub" in text             # Check substring

# Multiline strings
text = """
Multiple
lines
"""
```

## Lists (Mutable, Ordered)

```python
# Creation
items = [1, 2, 3]
items = list(range(5))    # [0, 1, 2, 3, 4]
empty = []

# Access
first = items[0]
last = items[-1]
subset = items[1:3]       # Slice [start:end]
subset = items[::2]       # Every 2nd element

# Modification
items.append(4)           # Add to end
items.insert(0, 0)        # Insert at index
items.extend([5, 6])      # Add multiple
items.remove(3)           # Remove first occurrence
popped = items.pop()      # Remove & return last
items.pop(0)              # Remove at index
items.clear()             # Remove all

# Other operations
len(items)
items.sort()              # In-place sort
sorted(items)             # Return new sorted list
items.reverse()
items.count(2)            # Count occurrences
items.index(3)            # Find index

# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]
```

## Dictionaries (Key-Value Pairs)

```python
# Creation
person = {"name": "Alice", "age": 30}
person = dict(name="Alice", age=30)
empty = {}

# Access
name = person["name"]              # KeyError if missing
name = person.get("name")          # None if missing
name = person.get("name", "Unknown")  # Default value

# Modification
person["email"] = "a@example.com"  # Add/update
del person["age"]                  # Delete
removed = person.pop("name")       # Remove & return

# Iteration
for key in person:
    print(key, person[key])
    
for key, value in person.items():
    print(key, value)
    
for key in person.keys():
    print(key)
    
for value in person.values():
    print(value)

# Other operations
len(person)
"name" in person          # Check key exists
person.update({"age": 31, "city": "NYC"})

# Dict comprehension
squares = {x: x**2 for x in range(5)}
```

## Sets (Unique, Unordered)

```python
# Creation
numbers = {1, 2, 3}
numbers = set([1, 2, 2, 3])  # Duplicates removed
empty = set()                 # NOT {}, that's a dict

# Operations
numbers.add(4)
numbers.remove(2)        # KeyError if missing
numbers.discard(2)       # No error if missing
numbers.clear()

# Set operations
a = {1, 2, 3}
b = {3, 4, 5}
a | b                    # Union: {1, 2, 3, 4, 5}
a & b                    # Intersection: {3}
a - b                    # Difference: {1, 2}
a ^ b                    # Symmetric diff: {1, 2, 4, 5}
```

## Tuples (Immutable, Ordered)

```python
# Creation
point = (10, 20)
single = (1,)            # Note the comma
coords = 10, 20, 30      # Parens optional

# Unpacking
x, y = point
x, y, z = coords
first, *rest = items     # Rest as list
```

## Control Flow

```python
# If-elif-else
if condition:
    pass
elif other_condition:
    pass
else:
    pass

# Ternary operator
value = "yes" if condition else "no"

# For loops
for item in items:
    print(item)

for i in range(5):       # 0 to 4
    print(i)

for i in range(2, 10, 2):  # Start, stop, step
    print(i)

for i, item in enumerate(items):
    print(f"{i}: {item}")

for key, value in dict.items():
    print(key, value)

for a, b in zip(list1, list2):
    print(a, b)

# While loops
while condition:
    pass

# Loop control
break                    # Exit loop
continue                 # Skip to next iteration

# For-else (runs if no break)
for item in items:
    if condition:
        break
else:
    print("No break occurred")
```

## Functions

```python
# Basic function
def greet(name):
    return f"Hello, {name}"

# With type hints
def add(a: int, b: int) -> int:
    return a + b

# Default arguments
def greet(name: str = "World") -> str:
    return f"Hello, {name}"

# Variable arguments
def sum_all(*args):
    return sum(args)

def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Combined
def func(required, *args, optional=None, **kwargs):
    pass

# Lambda (anonymous function)
square = lambda x: x**2
sorted_items = sorted(items, key=lambda x: x["name"])

# Docstring
def calculate(x: float, y: float) -> float:
    """
    Calculate something important.
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        The result
    """
    return x + y
```

## Classes

```python
# Basic class
class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    def greet(self) -> str:
        return f"Hi, I'm {self.name}"
    
    def __str__(self) -> str:
        return f"Person({self.name}, {self.age})"

# Inheritance
class Employee(Person):
    def __init__(self, name: str, age: int, salary: float):
        super().__init__(name, age)
        self.salary = salary

# Class and static methods
class MyClass:
    count = 0  # Class variable
    
    @classmethod
    def from_string(cls, data: str):
        # Alternative constructor
        return cls(*data.split(","))
    
    @staticmethod
    def helper(x: int) -> int:
        # No access to self or cls
        return x * 2

# Properties
class Circle:
    def __init__(self, radius: float):
        self._radius = radius
    
    @property
    def radius(self) -> float:
        return self._radius
    
    @radius.setter
    def radius(self, value: float):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self) -> float:
        return 3.14159 * self._radius ** 2

# Dataclass (Python 3.7+)
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0  # Default value
```

## Error Handling

```python
# Try-except
try:
    result = risky_operation()
except ValueError as e:
    print(f"Value error: {e}")
except (TypeError, KeyError):
    print("Type or Key error")
except Exception as e:
    print(f"Unexpected error: {e}")
else:
    print("No error occurred")
finally:
    print("Always executes")

# Raising exceptions
raise ValueError("Invalid input")
raise RuntimeError("Something went wrong")

# Custom exception
class CustomError(Exception):
    pass

# Context manager (suppress exceptions)
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove("file.txt")
```

## File Handling

```python
# Read file
with open("file.txt", "r") as f:
    content = f.read()           # Entire file
    lines = f.readlines()        # List of lines
    
with open("file.txt") as f:
    for line in f:               # Iterate lines
        print(line.strip())

# Write file
with open("file.txt", "w") as f:
    f.write("Hello\n")
    f.writelines(["Line 1\n", "Line 2\n"])

# Append
with open("file.txt", "a") as f:
    f.write("More content\n")

# JSON
import json

with open("data.json") as f:
    data = json.load(f)

with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

# Pathlib (modern approach)
from pathlib import Path

path = Path("folder") / "file.txt"
content = path.read_text()
path.write_text("content")
exists = path.exists()
is_file = path.is_file()
```

## Common Patterns

```python
# Enumerate (index + value)
for i, item in enumerate(items):
    print(f"{i}: {item}")

# Zip (parallel iteration)
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# Map (apply function to all)
squares = list(map(lambda x: x**2, numbers))

# Filter
evens = list(filter(lambda x: x % 2 == 0, numbers))

# Any/All
any(x > 10 for x in numbers)    # True if any match
all(x > 0 for x in numbers)     # True if all match

# Sorted with key
sorted_people = sorted(people, key=lambda p: p.age)
sorted_items = sorted(items, key=lambda x: x["name"])

# Reverse iteration
for item in reversed(items):
    print(item)

# Dictionary get with default
value = my_dict.get("key", default_value)

# Swap variables
a, b = b, a

# Check type
isinstance(obj, str)
isinstance(obj, (int, float))

# Walrus operator (Python 3.8+)
if (n := len(items)) > 10:
    print(f"Large list: {n} items")
```

## Useful Built-ins

```python
# Type conversion
int("42")
float("3.14")
str(42)
list("abc")          # ['a', 'b', 'c']
tuple([1, 2])

# Math
abs(-5)
round(3.7)
pow(2, 3)            # 2**3
min(1, 2, 3)
max(items)
sum([1, 2, 3])

# String/sequence
len(items)
sorted(items)
reversed(items)      # Iterator

# Functional
map(func, items)
filter(func, items)
zip(list1, list2)
enumerate(items)
```

## Imports

```python
# Basic import
import math
math.sqrt(16)

# Import specific items
from math import sqrt, pi
sqrt(16)

# Import with alias
import numpy as np
from matplotlib import pyplot as plt

# Import all (avoid in production)
from math import *

# Relative imports (in packages)
from . import module          # Same directory
from .. import module         # Parent directory
from .submodule import func   # Subdirectory
```

## Context Managers

```python
# With statement
with open("file.txt") as f:
    content = f.read()

# Custom context manager
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.time()
    try:
        yield
    finally:
        print(f"Took {time.time() - start}s")

with timer():
    # Code to time
    pass
```

## Decorators

```python
# Function decorator
def log_calls(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_calls
def greet(name):
    return f"Hello, {name}"

# With parameters
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("Hello")
```

## Generators

```python
# Generator function
def count_up_to(n):
    i = 0
    while i < n:
        yield i
        i += 1

# Generator expression
squares = (x**2 for x in range(10))

# Benefits: memory efficient for large sequences
```

## Common Modules

```python
# os - Operating system
import os
os.getcwd()              # Current directory
os.listdir("path")       # List files
os.path.exists("file")
os.path.join("dir", "file")

# sys - System specific
import sys
sys.argv                 # Command line args
sys.exit()              # Exit program

# datetime
from datetime import datetime, timedelta
now = datetime.now()
today = datetime.today()
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
parsed = datetime.strptime("2025-01-01", "%Y-%m-%d")
later = now + timedelta(days=7)

# random
import random
random.randint(1, 10)
random.choice(items)
random.shuffle(items)
random.sample(items, 3)

# collections
from collections import Counter, defaultdict, deque
counts = Counter([1, 2, 2, 3, 3, 3])
dd = defaultdict(list)   # No KeyError
queue = deque([1, 2, 3])

# itertools
from itertools import combinations, permutations, product
list(combinations([1, 2, 3], 2))
list(product([1, 2], ['a', 'b']))

# re - Regular expressions
import re
match = re.search(r'\d+', text)
matches = re.findall(r'\d+', text)
replaced = re.sub(r'\d+', 'X', text)
```

---

**Pro Tips:**
- Use type hints for better code documentation
- Prefer comprehensions over loops for simple transformations
- Always use `with` for file operations
- Use `Path` instead of `os.path` for modern code
- Check for `None` with `is None`, not `== None`
- Avoid mutable default arguments: use `def func(items=None):`
