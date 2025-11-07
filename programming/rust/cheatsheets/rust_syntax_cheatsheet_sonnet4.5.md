# Rust Syntax Cheat Sheet

A comprehensive reference for Rust programming language syntax.

## Variables and Mutability

```rust
// Immutable variable (default)
let x = 5;

// Mutable variable
let mut y = 10;
y = 15;

// Constant (must have type annotation)
const MAX_POINTS: u32 = 100_000;

// Shadowing
let x = 5;
let x = x + 1;
let x = "string"; // Can change type with shadowing
```

## Data Types

### Scalar Types

```rust
// Integers
let a: i8 = -128;          // 8-bit signed
let b: u8 = 255;           // 8-bit unsigned
let c: i32 = -2_147_483_648; // 32-bit signed (default)
let d: u64 = 18_446_744_073_709_551_615; // 64-bit unsigned

// Floating point
let x: f32 = 3.14;         // 32-bit float
let y: f64 = 2.71828;      // 64-bit float (default)

// Boolean
let t: bool = true;
let f: bool = false;

// Character (4 bytes, Unicode)
let c: char = 'z';
let emoji: char = '😻';
```

### Compound Types

```rust
// Tuple
let tup: (i32, f64, u8) = (500, 6.4, 1);
let (x, y, z) = tup;       // Destructuring
let first = tup.0;         // Index access

// Array (fixed size)
let arr: [i32; 5] = [1, 2, 3, 4, 5];
let arr = [3; 5];          // [3, 3, 3, 3, 3]
let first = arr[0];
```

## Functions

```rust
// Basic function
fn greet() {
    println!("Hello!");
}

// Function with parameters
fn add(x: i32, y: i32) -> i32 {
    x + y  // Expression (no semicolon) = return value
}

// Explicit return
fn subtract(x: i32, y: i32) -> i32 {
    return x - y;
}

// Multiple return values (tuple)
fn swap(x: i32, y: i32) -> (i32, i32) {
    (y, x)
}
```

## Control Flow

### If Expressions

```rust
// Basic if
if x > 5 {
    println!("Greater");
} else if x == 5 {
    println!("Equal");
} else {
    println!("Less");
}

// If as expression
let number = if condition { 5 } else { 6 };
```

### Loops

```rust
// Infinite loop
loop {
    println!("Forever!");
    break; // Exit loop
}

// Loop with return value
let result = loop {
    counter += 1;
    if counter == 10 {
        break counter * 2;
    }
};

// While loop
while number != 0 {
    println!("{}", number);
    number -= 1;
}

// For loop
for element in array.iter() {
    println!("{}", element);
}

// Range
for number in 1..4 {  // 1, 2, 3
    println!("{}", number);
}

for number in 1..=4 { // 1, 2, 3, 4
    println!("{}", number);
}
```

### Match

```rust
// Basic match
match value {
    1 => println!("One"),
    2 => println!("Two"),
    3 | 4 => println!("Three or Four"),
    _ => println!("Other"), // Default case
}

// Match with return
let result = match number {
    1 => "one",
    2 => "two",
    _ => "many",
};

// Match with ranges
match age {
    0..=12 => "child",
    13..=19 => "teen",
    _ => "adult",
}

// Match with guards
match number {
    x if x < 0 => "negative",
    x if x > 0 => "positive",
    _ => "zero",
}
```

## Ownership and Borrowing

```rust
// Ownership transfer (move)
let s1 = String::from("hello");
let s2 = s1; // s1 is no longer valid

// Clone (deep copy)
let s1 = String::from("hello");
let s2 = s1.clone();

// Immutable reference (borrowing)
let s1 = String::from("hello");
let len = calculate_length(&s1);

fn calculate_length(s: &String) -> usize {
    s.len()
}

// Mutable reference
let mut s = String::from("hello");
change(&mut s);

fn change(s: &mut String) {
    s.push_str(", world");
}

// Multiple immutable references OK
let r1 = &s;
let r2 = &s;

// Cannot have mutable and immutable references simultaneously
```

## Structs

```rust
// Define struct
struct User {
    username: String,
    email: String,
    age: u32,
    active: bool,
}

// Create instance
let user = User {
    email: String::from("user@example.com"),
    username: String::from("user123"),
    age: 25,
    active: true,
};

// Access fields
let email = user.email;

// Mutable instance
let mut user = User { /* ... */ };
user.email = String::from("new@example.com");

// Field init shorthand
fn build_user(email: String, username: String) -> User {
    User {
        email,
        username,
        age: 0,
        active: true,
    }
}

// Struct update syntax
let user2 = User {
    email: String::from("another@example.com"),
    ..user1
};

// Tuple structs
struct Color(i32, i32, i32);
let black = Color(0, 0, 0);

// Unit-like structs
struct AlwaysEqual;
let subject = AlwaysEqual;
```

### Methods

```rust
impl User {
    // Method
    fn is_adult(&self) -> bool {
        self.age >= 18
    }
    
    // Method with mutation
    fn increment_age(&mut self) {
        self.age += 1;
    }
    
    // Associated function (no self)
    fn new(email: String, username: String) -> User {
        User {
            email,
            username,
            age: 0,
            active: true,
        }
    }
}

// Usage
let user = User::new(email, username);
let adult = user.is_adult();
```

## Enums

```rust
// Basic enum
enum IpAddrKind {
    V4,
    V6,
}

let four = IpAddrKind::V4;

// Enum with data
enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

let home = IpAddr::V4(127, 0, 0, 1);
let loopback = IpAddr::V6(String::from("::1"));

// Enum methods
impl IpAddr {
    fn print(&self) {
        // implementation
    }
}

// Option enum (built-in)
let some_number: Option<i32> = Some(5);
let no_number: Option<i32> = None;

// Result enum (built-in)
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

## Pattern Matching with Enums

```rust
match coin {
    Coin::Penny => 1,
    Coin::Nickel => 5,
    Coin::Dime => 10,
    Coin::Quarter(state) => {
        println!("State: {:?}", state);
        25
    }
}

// if let (shorthand for single pattern)
if let Some(3) = some_value {
    println!("three");
}

// while let
while let Some(top) = stack.pop() {
    println!("{}", top);
}
```

## Collections

### Vector

```rust
// Create vector
let v: Vec<i32> = Vec::new();
let v = vec![1, 2, 3];

// Add elements
let mut v = Vec::new();
v.push(5);
v.push(6);

// Access elements
let third = &v[2];              // Panics if out of bounds
let third = v.get(2);           // Returns Option<&T>

// Iterate
for i in &v {
    println!("{}", i);
}

// Iterate with mutation
for i in &mut v {
    *i += 50;
}
```

### String

```rust
// Create string
let mut s = String::new();
let s = String::from("hello");
let s = "hello".to_string();

// Append
s.push_str(" world");
s.push('!');

// Concatenation
let s3 = s1 + &s2;              // s1 moved
let s = format!("{}-{}-{}", s1, s2, s3);

// Indexing (not allowed directly)
let hello = "Здравствуйте";
let s = &hello[0..4];           // Be careful with UTF-8!

// Iteration
for c in "नमस्ते".chars() {
    println!("{}", c);
}

for b in "नमस्ते".bytes() {
    println!("{}", b);
}
```

### HashMap

```rust
use std::collections::HashMap;

// Create
let mut scores = HashMap::new();

// Insert
scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);

// Access
let team_name = String::from("Blue");
let score = scores.get(&team_name);  // Returns Option<&V>

// Iterate
for (key, value) in &scores {
    println!("{}: {}", key, value);
}

// Update
scores.insert(String::from("Blue"), 25);  // Overwrite

// Insert if not exists
scores.entry(String::from("Blue")).or_insert(50);

// Update based on old value
let count = map.entry(word).or_insert(0);
*count += 1;
```

## Error Handling

### panic!

```rust
panic!("crash and burn");
```

### Result

```rust
use std::fs::File;
use std::io::ErrorKind;

// Basic Result handling
let f = File::open("hello.txt");

let f = match f {
    Ok(file) => file,
    Err(error) => panic!("Problem: {:?}", error),
};

// Match on error kind
let f = File::open("hello.txt").unwrap_or_else(|error| {
    if error.kind() == ErrorKind::NotFound {
        File::create("hello.txt").unwrap_or_else(|error| {
            panic!("Problem creating file: {:?}", error);
        })
    } else {
        panic!("Problem opening file: {:?}", error);
    }
});

// Shortcuts
let f = File::open("hello.txt").unwrap();  // Panic on error
let f = File::open("hello.txt").expect("Failed to open"); // Custom panic message

// Propagating errors with ?
fn read_username() -> Result<String, io::Error> {
    let mut f = File::open("hello.txt")?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}

// ? can be chained
fn read_username() -> Result<String, io::Error> {
    let mut s = String::new();
    File::open("hello.txt")?.read_to_string(&mut s)?;
    Ok(s)
}
```

## Generics

```rust
// Generic function
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}

// Generic struct
struct Point<T> {
    x: T,
    y: T,
}

// Multiple type parameters
struct Point<T, U> {
    x: T,
    y: U,
}

// Generic enum
enum Option<T> {
    Some(T),
    None,
}

// Generic methods
impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
}

// Specific type implementation
impl Point<f32> {
    fn distance_from_origin(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}
```

## Traits

```rust
// Define trait
trait Summary {
    fn summarize(&self) -> String;
    
    // Default implementation
    fn default_summary(&self) -> String {
        String::from("(Read more...)")
    }
}

// Implement trait
impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}, by {}", self.headline, self.author)
    }
}

// Trait as parameter
fn notify(item: &impl Summary) {
    println!("Breaking news! {}", item.summarize());
}

// Trait bound syntax
fn notify<T: Summary>(item: &T) {
    println!("Breaking news! {}", item.summarize());
}

// Multiple trait bounds
fn notify<T: Summary + Display>(item: &T) { }

// Where clause (cleaner for complex bounds)
fn some_function<T, U>(t: &T, u: &U) -> i32
    where T: Display + Clone,
          U: Clone + Debug
{ }

// Return types that implement traits
fn returns_summarizable() -> impl Summary {
    Tweet { /* ... */ }
}

// Conditional trait implementation
impl<T: Display> ToString for T {
    // implementation
}
```

### Common Traits

```rust
// Clone - explicit copy
#[derive(Clone)]
struct MyStruct;

// Copy - implicit copy (for stack-only data)
#[derive(Copy, Clone)]
struct Point { x: i32, y: i32 }

// Debug - formatting with {:?}
#[derive(Debug)]
struct Rectangle { width: u32, height: u32 }

// PartialEq, Eq - equality comparison
#[derive(PartialEq, Eq)]
struct Person { name: String }

// PartialOrd, Ord - ordering
#[derive(PartialOrd, Ord)]
struct Height(i32);

// Derive multiple traits
#[derive(Debug, Clone, PartialEq)]
struct User { name: String }
```

## Lifetimes

```rust
// Lifetime annotation in function signature
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

// Lifetime in struct
struct ImportantExcerpt<'a> {
    part: &'a str,
}

// Lifetime in methods
impl<'a> ImportantExcerpt<'a> {
    fn announce_and_return_part(&self, announcement: &str) -> &str {
        println!("Attention: {}", announcement);
        self.part
    }
}

// Static lifetime
let s: &'static str = "I live for the entire program";
```

## Closures

```rust
// Basic closure
let add_one = |x| x + 1;
let result = add_one(5);

// Closure with type annotations
let add_one = |x: i32| -> i32 { x + 1 };

// Capturing environment
let x = 4;
let equal_to_x = |z| z == x;

// Move closure (takes ownership)
let x = vec![1, 2, 3];
let equal_to_x = move |z| z == x;

// Fn traits
// FnOnce - consumes captured variables
// FnMut - mutably borrows
// Fn - immutably borrows
```

## Iterators

```rust
// Create iterator
let v = vec![1, 2, 3];
let iter = v.iter();

// Consuming adaptors
let total: i32 = v.iter().sum();

// Iterator adaptors
let v2: Vec<_> = v.iter().map(|x| x + 1).collect();

// Filter
let even: Vec<_> = v.iter().filter(|x| *x % 2 == 0).collect();

// Chaining
let result: i32 = v.iter()
    .filter(|x| *x > 2)
    .map(|x| x * 2)
    .sum();

// Custom iterator
struct Counter {
    count: u32,
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < 5 {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}
```

## Modules and Crates

```rust
// Define module
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
        fn seat_at_table() {}
    }
}

// Use keyword
use crate::front_of_house::hosting;
use std::collections::HashMap;

// Use with alias
use std::io::Result as IoResult;

// Glob operator
use std::collections::*;

// Re-exporting
pub use crate::front_of_house::hosting;

// Nested paths
use std::{cmp::Ordering, io};
use std::io::{self, Write};

// External crate (in Cargo.toml)
// [dependencies]
// rand = "0.8.5"

use rand::Rng;
```

## Smart Pointers

### Box

```rust
// Heap allocation
let b = Box::new(5);

// Recursive type
enum List {
    Cons(i32, Box<List>),
    Nil,
}

let list = Cons(1, Box::new(Cons(2, Box::new(Nil))));
```

### Rc (Reference Counted)

```rust
use std::rc::Rc;

let a = Rc::new(5);
let b = Rc::clone(&a);
let c = Rc::clone(&a);

println!("count: {}", Rc::strong_count(&a));
```

### RefCell

```rust
use std::cell::RefCell;

let x = RefCell::new(5);
*x.borrow_mut() += 1;
```

## Concurrency

### Threads

```rust
use std::thread;
use std::time::Duration;

// Spawn thread
let handle = thread::spawn(|| {
    for i in 1..10 {
        println!("thread: {}", i);
        thread::sleep(Duration::from_millis(1));
    }
});

handle.join().unwrap();

// Move closure
let v = vec![1, 2, 3];
let handle = thread::spawn(move || {
    println!("vector: {:?}", v);
});
```

### Channels

```rust
use std::sync::mpsc;

// Create channel
let (tx, rx) = mpsc::channel();

// Send
thread::spawn(move || {
    let val = String::from("hi");
    tx.send(val).unwrap();
});

// Receive
let received = rx.recv().unwrap();

// Multiple producers
let (tx, rx) = mpsc::channel();
let tx1 = tx.clone();

// Iterate over received values
for received in rx {
    println!("Got: {}", received);
}
```

### Mutex

```rust
use std::sync::Mutex;

let m = Mutex::new(5);

{
    let mut num = m.lock().unwrap();
    *num = 6;
}

// Arc (Atomic Reference Counted) for thread-safe sharing
use std::sync::Arc;

let counter = Arc::new(Mutex::new(0));
let counter_clone = Arc::clone(&counter);

thread::spawn(move || {
    let mut num = counter_clone.lock().unwrap();
    *num += 1;
});
```

## Macros

```rust
// Macro invocation
println!("Hello, {}!", name);
vec![1, 2, 3];

// Declarative macro
macro_rules! vec {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            temp_vec
        }
    };
}
```

## Testing

```rust
// Unit test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
    
    #[test]
    fn another() {
        assert!(true);
    }
    
    #[test]
    #[should_panic]
    fn it_panics() {
        panic!("This should panic");
    }
    
    #[test]
    fn with_result() -> Result<(), String> {
        if 2 + 2 == 4 {
            Ok(())
        } else {
            Err(String::from("two plus two does not equal four"))
        }
    }
}

// Run tests
// cargo test
// cargo test test_name
// cargo test -- --show-output
```

## Attributes

```rust
// Conditional compilation
#[cfg(target_os = "linux")]
fn are_you_on_linux() {
    println!("You are running linux!");
}

// Allow/deny lints
#[allow(dead_code)]
#[deny(warnings)]

// Deprecated
#[deprecated(since = "1.0.0", note = "use new_function instead")]
fn old_function() {}

// Documentation
/// This is a doc comment
/// It supports markdown
/// 
/// # Examples
/// ```
/// let result = my_function(5);
/// assert_eq!(result, 10);
/// ```
fn my_function(x: i32) -> i32 {
    x * 2
}
```

## Common Operators

```rust
// Arithmetic
+ - * / %

// Comparison
== != < > <= >=

// Logical
&& || !

// Bitwise
& | ^ << >>

// Assignment
= += -= *= /= %=

// Range
.. ..= // (exclusive) (inclusive)

// Dereference
*

// Reference
& &mut

// Type cast
as

// Pattern matching
match | @

// Namespace
::
```

## Common Commands

```bash
# Create new project
cargo new project_name
cargo new --lib library_name

# Build project
cargo build
cargo build --release

# Run project
cargo run

# Check compilation
cargo check

# Test
cargo test

# Documentation
cargo doc --open

# Update dependencies
cargo update

# Format code
cargo fmt

# Lint
cargo clippy
```

## Useful Patterns

### Option Handling

```rust
// unwrap_or
let value = option.unwrap_or(default_value);

// unwrap_or_else
let value = option.unwrap_or_else(|| expensive_computation());

// map
let value = option.map(|x| x * 2);

// and_then
let value = option.and_then(|x| Some(x * 2));

// ok_or
let result: Result<T, E> = option.ok_or(error);
```

### Result Handling

```rust
// unwrap_or
let value = result.unwrap_or(default_value);

// unwrap_or_else
let value = result.unwrap_or_else(|err| handle_error(err));

// map
let value = result.map(|x| x * 2);

// and_then
let value = result.and_then(|x| Ok(x * 2));

// ok
let option: Option<T> = result.ok();
```

## Type Aliases

```rust
type Kilometers = i32;
type Result<T> = std::result::Result<T, std::io::Error>;
```

## Constants vs Statics

```rust
// Constant (inlined at compile time)
const MAX_POINTS: u32 = 100_000;

// Static (fixed memory location)
static HELLO_WORLD: &str = "Hello, world!";

// Mutable static (unsafe)
static mut COUNTER: u32 = 0;
```

---

**Note**: This cheat sheet covers Rust fundamentals. For more detailed information, refer to the [official Rust documentation](https://doc.rust-lang.org/).
