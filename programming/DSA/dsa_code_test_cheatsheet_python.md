# DSA for Coding Tests — Python Cheat Sheet
_Last updated: 2025-11-04_

> Quick patterns, tiny proofs of life, and tasty time-complexities. Pack light, pass tests.

---

## 1) Big-O at a Glance
- **Common**: `O(1) < O(log n) < O(n) < O(n log n) < O(n^2) < O(2^n) < O(n!)`
- **Rules of thumb**
  - Drop constants; keep dominant terms.
  - Worst-case unless told otherwise.
  - Space ≠ Time: trading memory for speed is often worth it.

**Python gotchas**
- `list.append`: amortized `O(1)`
- `pop()`: tail `O(1)`, head `O(n)`
- `in` on list: `O(n)`; on set/dict: average `O(1)`
- `heapq`: min-heap, push/pop `O(log n)`
- `sorted`: Timsort `O(n log n)`

---

## 2) Core Data Structures
### Arrays & Strings
- Use two pointers, sliding window, prefix/suffix, frequency arrays.
```python
# Reverse words in string
s = "hello world"
" ".join(reversed(s.split()))
```

### Hash Map / Counter / Set
```python
from collections import Counter, defaultdict
cnt = Counter(s)                 # freq map
seen = set()
d = defaultdict(int)             # auto-0
```

### Stack (LIFO)
```python
# Valid parentheses
def valid_parens(s: str) -> bool:
    pairs = {')':'(', ']':'[', '}':'{'}
    st = []
    for c in s:
        if c in '([{': st.append(c)
        elif not st or st.pop() != pairs[c]: return False
    return not st
```

### Queue / Deque
```python
from collections import deque
q = deque()
q.append(x); q.popleft()
```

### Heap (Priority Queue)
```python
import heapq
# K smallest
def k_smallest(nums, k):
    return heapq.nsmallest(k, nums)  # or use heapq heapify/push/pop
```

### Linked List (patterns)
- Fast/slow pointers (cycle, middle, kth from end)
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next; fast = fast.next.next
        if slow is fast: return True
    return False
```

### Trees (Binary / BST)
- Inorder = sorted for BST; DFS (pre/in/post), BFS (level order)
```python
def inorder(root):
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def bst_search(root, x):
    cur = root
    while cur:
        if x == cur.val: return cur
        cur = cur.left if x < cur.val else cur.right
    return None
```

### Graphs
- Represent with adjacency list: `g = defaultdict(list)`
- Traversals: BFS (shortest path in unweighted), DFS (components/topology)
```python
from collections import deque, defaultdict
def bfs(g, start):
    vis = {start}; q = deque([start]); order = []
    while q:
        u = q.popleft(); order.append(u)
        for v in g[u]:
            if v not in vis:
                vis.add(v); q.append(v)
    return order
```

---

## 3) Recurring Algorithm Patterns
### Two Pointers
```python
# 2-sum sorted
def two_sum_sorted(a, target):
    i, j = 0, len(a)-1
    while i < j:
        s = a[i] + a[j]
        if s == target: return [i, j]
        if s < target: i += 1
        else: j -= 1
```

### Sliding Window
```python
# Longest substring without repeating characters
def lengthOfLongestSubstring(s):
    seen = {}
    left = ans = 0
    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1
        seen[ch] = right
        ans = max(ans, right - left + 1)
    return ans
```

### Prefix Sum / Difference Array
```python
# Subarray sum equals k
def subarray_sum(nums, k):
    from collections import defaultdict
    pref = 0; cnt = defaultdict(int); cnt[0] = 1; ans = 0
    for x in nums:
        pref += x
        ans += cnt[pref - k]
        cnt[pref] += 1
    return ans
```

### Binary Search (on index / on answer)
```python
# Index
def bin_search(a, x):
    lo, hi = 0, len(a)-1
    while lo <= hi:
        mid = (lo+hi)//2
        if a[mid] < x: lo = mid+1
        elif a[mid] > x: hi = mid-1
        else: return mid
    return -1

# On answer (e.g., minimize capacity)
def feasible(cap, weights, days):
    d = 1; cur = 0
    for w in weights:
        if w > cap: return False
        if cur + w > cap: d += 1; cur = 0
        cur += w
    return d <= days

def min_capacity(weights, days):
    lo, hi = max(weights), sum(weights)
    while lo < hi:
        mid = (lo+hi)//2
        if feasible(mid, weights, days): hi = mid
        else: lo = mid+1
    return lo
```

### Greedy
- Exchange argument, matroid-like structure, local-optimal → global-optimal.
```python
# Activity selection (max non-overlapping intervals)
def max_non_overlapping(intervals):
    intervals.sort(key=lambda x: x[1])
    ans = 0; end = float('-inf')
    for s,e in intervals:
        if s >= end:
            ans += 1; end = e
    return ans
```

### Divide & Conquer / Merge Sort
```python
def merge_sort(a):
    if len(a) <= 1: return a
    mid = len(a)//2
    L, R = merge_sort(a[:mid]), merge_sort(a[mid:])
    i = j = 0; out = []
    while i < len(L) and j < len(R):
        if L[i] <= R[j]: out.append(L[i]); i += 1
        else: out.append(R[j]); j += 1
    out.extend(L[i:]); out.extend(R[j:])
    return out
```

### Backtracking
```python
# Subsets
def subsets(nums):
    ans, path = [], []
    def dfs(i):
        if i == len(nums): ans.append(path[:]); return
        path.append(nums[i]); dfs(i+1)
        path.pop(); dfs(i+1)
    dfs(0); return ans
```

### Dynamic Programming (DP)
- Table or memo. State design matters most.
```python
# 0/1 Knapsack (value maximization)
def knap(values, weights, W):
    n = len(values)
    dp = [0]*(W+1)
    for i in range(n):
        for w in range(W, weights[i]-1, -1):
            dp[w] = max(dp[w], dp[w-weights[i]] + values[i])
    return dp[W]

# Edit distance
def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
```

---

## 4) Graph Algorithms
```python
# Topological sort (Kahn)
from collections import deque, defaultdict
def topo_sort(n, edges):
    g = defaultdict(list); indeg = [0]*n
    for u,v in edges:
        g[u].append(v); indeg[v] += 1
    q = deque([i for i in range(n) if indeg[i]==0])
    order = []
    while q:
        u = q.popleft(); order.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v]==0: q.append(v)
    return order if len(order)==n else []  # cycle → empty

# Dijkstra (non-negative weights)
import heapq
def dijkstra(n, g, src):
    dist = [float('inf')]*n; dist[src] = 0
    pq = [(0, src)]
    while pq:
        d,u = heapq.heappop(pq)
        if d!=dist[u]: continue
        for v,w in g[u]:
            if dist[v] > d + w:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist
```

---

## 5) Sorting & Selection
```python
# Quickselect: kth smallest (0-indexed)
import random
def quickselect(a, k):
    l, r = 0, len(a)-1
    while True:
        pivot = a[random.randint(l, r)]
        L = [x for x in a[l:r+1] if x < pivot]
        E = [x for x in a[l:r+1] if x == pivot]
        G = [x for x in a[l:r+1] if x > pivot]
        if k < len(L): a[l:r+1] = L + E + G; r = l + len(L) - 1
        elif k < len(L) + len(E): return pivot
        else: 
            k -= len(L) + len(E)
            a[l:r+1] = L + E + G; l = r - len(G) + 1
```

---

## 6) Bit Tricks (handy)
```python
# Check ith bit
(x >> i) & 1
# Set/Clear/Toggle ith bit
x | (1<<i), x & ~(1<<i), x ^ (1<<i)
# Lowest set bit
x & -x
# Count bits
bin(x).count("1")
```

---

## 7) Template Snippets
```python
# Fast I/O (CP style)
import sys
data = sys.stdin.read().strip().split()
it = iter(data)
# n = int(next(it))

# Disjoint Set Union (Union-Find)
class DSU:
    def __init__(self, n):
        self.p = list(range(n)); self.r = [0]*n
    def find(self, x):
        if self.p[x] != x: self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b: return False
        if self.r[a] < self.r[b]: a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]: self.r[a] += 1
        return True
```

---

## 8) Strategy on the Day
- Clarify constraints, input size, and edge cases.
- Start with a brute force; improve with patterns (window, hash, binary search, DP).
- Prove correctness in 1–2 sentences while coding.
- Add quick tests for edge cases (empty, single, extremes).
- If stuck: reduce to a known pattern or solve a simpler subproblem first.

**Good luck — and may your bugs be off-by-none.**
