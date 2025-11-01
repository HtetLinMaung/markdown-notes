# Python Generators - Complete Guide

## The Core Concept

**Generators are functions that can pause and resume, yielding values one at a time instead of returning all at once.**

### `return` vs `yield`

- **`return`**: "Here's your answer. I'm done. Goodbye."
- **`yield`**: "Here's a value. I'll pause here. Call me again when you want the next one."

## Basic Example

### Regular Function (with `return`)
```python
def get_numbers():
    numbers = [1, 2, 3]
    return numbers

result = get_numbers()
print(result)  # [1, 2, 3] - entire list at once
```

### Generator Function (with `yield`)
```python
def generate_numbers():
    yield 1
    yield 2
    yield 3

result = generate_numbers()
print(result)  # <generator object> - not the values yet!

# Get values one at a time
print(next(result))  # 1 - first yield
print(next(result))  # 2 - second yield
print(next(result))  # 3 - third yield
print(next(result))  # StopIteration error - no more yields
```

**What happens:**
1. Calling `generate_numbers()` doesn't run the function - it creates a generator object
2. Each `next()` call runs until it hits a `yield`
3. The function **pauses** at each `yield` and **remembers where it was**
4. Next call continues from where it paused

## Most Common Usage: For Loops

You rarely call `next()` manually. Use loops instead:

```python
def generate_numbers():
    yield 1
    yield 2
    yield 3

# Loop automatically calls next() until exhausted
for num in generate_numbers():
    print(num)
# Output:
# 1
# 2
# 3
```

## Why Generators Matter: Memory Efficiency

### ❌ Without Generator (uses lots of memory)
```python
def get_million_numbers():
    numbers = []
    for i in range(1_000_000):
        numbers.append(i * 2)
    return numbers  # Creates entire list in memory!

# All million numbers stored in memory at once
for num in get_million_numbers():
    print(num)
```

### ✅ With Generator (memory efficient)
```python
def generate_million_numbers():
    for i in range(1_000_000):
        yield i * 2  # Generate one at a time

# Only one number in memory at a time!
for num in generate_million_numbers():
    print(num)
```

**Key difference:** Generators create values on-demand, not all at once.

## How `yield` Works: The Bookmark Analogy

Think of generators as functions with "bookmarks":

```python
def count_to_three():
    print("Starting")
    yield 1          # Bookmark 1: pause here
    print("After 1")
    yield 2          # Bookmark 2: pause here
    print("After 2")
    yield 3          # Bookmark 3: pause here
    print("Done")

gen = count_to_three()

print(next(gen))
# Output:
# Starting
# 1

print(next(gen))
# Output:
# After 1
# 2

print(next(gen))
# Output:
# After 2
# 3

print(next(gen))
# Output:
# Done
# StopIteration error
```

The function **remembers**:
- Where it stopped (which line)
- All local variables
- Where to continue next time

## Practical Examples

### 1. Infinite Sequences
```python
def infinite_counter():
    num = 0
    while True:
        yield num
        num += 1

counter = infinite_counter()
print(next(counter))  # 0
print(next(counter))  # 1
print(next(counter))  # 2
# ... continues forever

# Use with limit
counter = infinite_counter()
for i, num in enumerate(counter):
    if i >= 5:
        break
    print(num)  # Prints 0, 1, 2, 3, 4
```

### 2. File Processing (Memory Efficient)
```python
def read_large_file(filepath):
    """Process huge files without loading all into memory"""
    with open(filepath) as f:
        for line in f:
            yield line.strip()

# Process gigabyte file efficiently
for line in read_large_file("huge.txt"):
    process(line)
```

### 3. Fibonacci Sequence
```python
def fibonacci(limit):
    a, b = 0, 1
    count = 0
    while count < limit:
        yield a
        a, b = b, a + b
        count += 1

# Get first 10 Fibonacci numbers
for num in fibonacci(10):
    print(num)
# Output: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
```

### 4. Data Pipeline (Chaining Generators)
```python
def get_numbers():
    for i in range(10):
        yield i

def square_numbers(numbers):
    for num in numbers:
        yield num ** 2

def filter_evens(numbers):
    for num in numbers:
        if num % 2 == 0:
            yield num

# Chain generators together
pipeline = filter_evens(square_numbers(get_numbers()))
for num in pipeline:
    print(num)
# Output: 0, 4, 16, 36, 64 (even squares)
```

### 5. Batch Processing
```python
def batch(items, size=10):
    """Split items into batches of specified size"""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:  # Don't forget remaining items
        yield batch

# Use it
for group in batch(range(25), size=10):
    print(group)
# Output:
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# [20, 21, 22, 23, 24]
```

### 6. CSV Processing
```python
def parse_csv(filepath):
    """Parse CSV file line by line"""
    with open(filepath) as f:
        headers = next(f).strip().split(',')
        for line in f:
            values = line.strip().split(',')
            yield dict(zip(headers, values))

# Process large CSV without loading all into memory
for row in parse_csv('data.csv'):
    print(row)  # {'name': 'Alice', 'age': '30', ...}
```

### 7. API Pagination
```python
def fetch_all_pages(api_url):
    """Fetch all pages from paginated API"""
    page = 1
    while True:
        response = requests.get(f"{api_url}?page={page}")
        data = response.json()
        
        if not data:  # No more data
            break
            
        for item in data:
            yield item
        
        page += 1

# Use it
for item in fetch_all_pages('https://api.example.com/items'):
    process(item)
```

## Generator Expressions

Like list comprehensions, but with parentheses:

```python
# List comprehension (creates entire list)
squares_list = [x**2 for x in range(1000000)]  # Uses lots of memory

# Generator expression (creates values on-demand)
squares_gen = (x**2 for x in range(1000000))   # Uses almost no memory

# Use in loops
for square in squares_gen:
    print(square)

# Use with functions that accept iterables
total = sum(x**2 for x in range(1000))
max_value = max(x for x in range(100) if x % 2 == 0)
```

## Common Patterns

### Pattern 1: Transform Stream
```python
def transform_data(items):
    """Apply transformation to each item"""
    for item in items:
        processed = expensive_operation(item)
        yield processed
```

### Pattern 2: Filter Stream
```python
def filter_valid(items):
    """Keep only valid items"""
    for item in items:
        if is_valid(item):
            yield item
```

### Pattern 3: Flatten Nested Structure
```python
def flatten(nested_list):
    """Flatten nested lists"""
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)  # yield from = yield each item
        else:
            yield item

nested = [1, [2, 3, [4, 5]], 6]
print(list(flatten(nested)))  # [1, 2, 3, 4, 5, 6]
```

### Pattern 4: Window/Sliding Window
```python
def sliding_window(items, size=2):
    """Create sliding window over items"""
    window = []
    for item in items:
        window.append(item)
        if len(window) == size:
            yield tuple(window)
            window.pop(0)

for pair in sliding_window([1, 2, 3, 4, 5], size=2):
    print(pair)
# Output:
# (1, 2)
# (2, 3)
# (3, 4)
# (4, 5)
```

## Advanced: `yield from`

Delegate to another generator:

```python
def generator1():
    yield 1
    yield 2

def generator2():
    yield 3
    yield 4

def combined():
    # Old way
    for value in generator1():
        yield value
    for value in generator2():
        yield value
    
    # Better way with yield from
    yield from generator1()
    yield from generator2()

for num in combined():
    print(num)  # 1, 2, 3, 4
```

## Sending Values to Generators

Generators can receive values using `.send()`:

```python
def echo():
    while True:
        received = yield  # Can receive value here
        print(f"Received: {received}")

gen = echo()
next(gen)  # Prime the generator (run until first yield)
gen.send("Hello")   # Output: Received: Hello
gen.send("World")   # Output: Received: World
```

More practical example:

```python
def running_average():
    total = 0
    count = 0
    while True:
        value = yield total / count if count else 0
        total += value
        count += 1

avg = running_average()
next(avg)  # Prime it
print(avg.send(10))  # 10.0
print(avg.send(20))  # 15.0
print(avg.send(30))  # 20.0
```

## Mental Models

### Model 1: Pause Button
```python
def counter():
    print("Start")
    yield 1        # ⏸️ Pause here, return 1
    print("Middle")
    yield 2        # ⏸️ Pause here, return 2
    print("End")
```

### Model 2: Factory Assembly Line
- **Regular function**: Build all products, then ship everything
- **Generator**: Build one product, ship it, build next one when requested

### Model 3: Lazy Evaluation
```python
# Eager evaluation (all at once)
numbers = [expensive_calc(i) for i in range(1000)]  # Runs immediately

# Lazy evaluation (on-demand)
numbers = (expensive_calc(i) for i in range(1000))  # Runs when needed
```

## When to Use Generators

### ✅ Use Generators When:
- Processing large datasets (files, databases, APIs)
- Working with infinite sequences
- Creating data pipelines/transformations
- You don't need all values at once
- Memory efficiency is important
- Reading streams of data

### ❌ Don't Use Generators When:
- You need to access values multiple times (generators are one-time use)
- You need random access (like `items[5]`)
- You need to know the length beforehand (`len()` doesn't work)
- Working with small datasets where memory doesn't matter
- You need to sort or reverse the data

## Important Limitations

### 1. Single Use Only
```python
my_gen = (x for x in [1, 2, 3])
print(sum(my_gen))  # 6
print(sum(my_gen))  # 0 - exhausted! Need to create new generator
```

### 2. No Random Access
```python
my_gen = (x for x in range(10))
# my_gen[5]  # TypeError - can't index generators
```

### 3. No Length
```python
my_gen = (x for x in range(10))
# len(my_gen)  # TypeError - generators don't have length
```

### 4. Can't Go Backwards
```python
my_gen = (x for x in [1, 2, 3])
next(my_gen)  # 1
next(my_gen)  # 2
# Can't go back to 1
```

## Comparison: List vs Generator

| Feature | List | Generator |
|---------|------|-----------|
| Memory | Stores all items | Generates on-demand |
| Speed | Fast access | Slower per item |
| Reusability | Multiple iterations | Single iteration |
| Indexing | `items[5]` works | Not supported |
| Length | `len(items)` works | Not supported |
| When to use | Small datasets, need reuse | Large datasets, one-time use |

## Practice Exercises

Try implementing these generators:

```python
# 1. Range with step
def my_range(start, stop, step=1):
    current = start
    while current < stop:
        yield current
        current += step

# 2. Repeat items n times
def repeat_each(items, n):
    for item in items:
        for _ in range(n):
            yield item

# 3. Prime numbers
def primes():
    num = 2
    while True:
        is_prime = True
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            yield num
        num += 1

# 4. Read file in chunks
def read_chunks(filepath, chunk_size=1024):
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

# 5. Permutations
def permutations(items):
    if len(items) <= 1:
        yield items
    else:
        for i, item in enumerate(items):
            rest = items[:i] + items[i+1:]
            for p in permutations(rest):
                yield [item] + p
```

## Key Takeaways

1. **`yield` pauses and remembers**, `return` finishes and forgets
2. **Generators are lazy** - they compute values only when needed
3. **Use generators for memory efficiency** with large datasets
4. **Generators are single-use** - can't iterate twice
5. **Common with files, APIs, and streams** - process data without loading all at once
6. **Generator expressions** use `()` instead of `[]`
7. **Think "assembly line"** - produce items one at a time, not in bulk

## Additional Resources

- **Built-in generator functions**: `range()`, `enumerate()`, `zip()`, `map()`, `filter()`
- **Module**: `itertools` - powerful generator utilities
- **PEP 255**: Simple Generators (original proposal)
- **PEP 342**: Coroutines via Enhanced Generators