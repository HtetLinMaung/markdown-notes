
# Rust Syntax Cheatsheet

A fast, no-fluff refresher of the essentials (with tiny examples you can copy‑paste).

---

## Basics

```rust
fn main() {
    // Comments
    // Line comment
    /* Block comment */

    // Print
    println!("hello, {}!", "world");

    // Variables
    let x = 5;              // immutable by default
    let mut y = 10;         // mutable
    y += 1;

    // Constants
    const MAX_POINTS: u32 = 100_000;

    // Shadowing (rebind with new type or value)
    let spaces = "   ";
    let spaces = spaces.len();
}
```

### Primitive Types
```rust
// Integers: i8, i16, i32, i64, i128, isize; Unsigned: u8..u128, usize
let a: i32 = -1;
let b: u64 = 42;

// Floats: f32, f64
let pi: f64 = 3.1415;

// Bool & Char
let ok: bool = true;
let heart: char = '❤';
```

### Tuples & Arrays & Slices
```rust
let tup: (i32, bool, &str) = (42, true, "hi");
let (n, t, s) = tup;
println!("{}", tup.0); // 42

let arr = [1, 2, 3, 4];
let slice: &[i32] = &arr[1..3]; // [2, 3]
```

### Strings
```rust
let s: &str = "literal";             // string slice
let mut w = String::from("hello");   // owned, growable
w.push_str(" world");
let len = w.len();
let first = &w[0..1]; // slicing by bytes; beware UTF‑8 boundaries
```

---

## Functions, Expressions, Control Flow

```rust
fn add(a: i32, b: i32) -> i32 {
    a + b // expression (no semicolon returns this value)
}

fn main() {
    let n = if add(1,2) > 2 { 10 } else { 0 };
    for i in 0..3 { println!("{i}"); }       // 0,1,2
    let mut i = 0;
    while i < 3 { i += 1; }
    loop { break; }                          // infinite until break
}
```

---

## Ownership, Borrowing, Lifetimes (mini)
- **Move**: assignment transfers ownership.
- **Borrow**: `&T` (shared, read‑only), `&mut T` (exclusive, write).
- **Rules**: any number of shared borrows OR exactly one mutable borrow at a time.
- **Lifetimes**: names that prove references outlive their use.

```rust
fn takes_ownership(s: String) { println!("{s}"); }
fn borrows(s: &String) { println!("{}", s.len()); }

fn main() {
    let s = String::from("hi");
    borrows(&s);           // ok (borrow)
    // takes_ownership(s); // would move s
    takes_ownership(s.clone()); // keep original by cloning
}
```

```rust
// Lifetimes in signatures (common pattern)
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

---

## Structs, Enums, Methods

```rust
struct User { id: u64, name: String }

impl User {
    fn new(id: u64, name: impl Into<String>) -> Self {
        Self { id, name: name.into() }
    }
    fn rename(&mut self, to: &str) { self.name = to.into(); }
}

enum Shape {
    Circle { r: f64 },
    Rect { w: f64, h: f64 },
}

impl Shape {
    fn area(&self) -> f64 {
        match self {
            Shape::Circle { r } => std::f64::consts::PI * r * r,
            Shape::Rect { w, h } => w * h,
        }
    }
}
```

---

## Pattern Matching & `match`

```rust
let x = Some(3);
match x {
    Some(n) if n % 2 == 1 => println!("odd {n}"),
    Some(n) => println!("evenish {n}"),
    None => println!("no value"),
}

let pair = (0, 1);
let (a, b) = pair; // destructuring
```

Shorthand with `if let`/`while let`:
```rust
if let Some(v) = x { println!("{v}"); }
while let Some(ch) = "abc".chars().next() { break; }
```

---

## Traits, Generics, `impl`

```rust
trait Summary { fn summary(&self) -> String; }

impl Summary for User {
    fn summary(&self) -> String { format!("#{} {}", self.id, self.name) }
}

fn show<T: Summary>(x: &T) { println!("{}", x.summary()); }
// or: fn show<T>(x: &T) where T: Summary { ... }

// Blanket impl example
impl<T: ToString> Summary for T {
    fn summary(&self) -> String { self.to_string() }
}
```

---

## Collections & Iterators

```rust
use std::collections::{VecDeque, HashMap, HashSet};

let mut v = vec![1, 2, 3];
v.push(4);
let doubled: Vec<_> = v.iter().map(|x| x * 2).collect();

let mut map = HashMap::new();
map.insert("a", 1);
if let Some(v) = map.get("a") { println!("{v}"); }
for (k, v) in &map { println!("{k}={v}"); }
```

Common iterator adapters:
- `map`, `filter`, `filter_map`, `fold`, `sum`, `any`, `all`, `take`, `skip`, `enumerate`, `flat_map`

---

## `Option` & `Result`

```rust
fn maybe_div(a: i32, b: i32) -> Option<i32> {
    if b == 0 { None } else { Some(a / b) }
}

use std::fs;
use std::io;

fn read_text(path: &str) -> Result<String, io::Error> {
    let contents = fs::read_to_string(path)?; // `?` propagates errors
    Ok(contents)
}
```

Transformers:
- `opt.map(...)`, `opt.and_then(...)`
- `res.map(...)`, `res.map_err(...)`, `res.and_then(...)`
- `unwrap`, `expect` (panic!), `unwrap_or`, `unwrap_or_else`

---

## Error Handling Quick Patterns

```rust
// Custom error type
use thiserror::Error;

#[derive(Debug, Error)]
enum AppError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad input: {0}")]
    BadInput(String),
}

type Result<T> = std::result::Result<T, AppError>;

fn run() -> Result<()> {
    let txt = std::fs::read_to_string("file.txt")?;
    if txt.is_empty() { return Err(AppError::BadInput("empty".into())); }
    Ok(())
}
```

*No `thiserror`? Use plain enums and `From` impls, or `anyhow` for quick prototypes.*

---

## Closures

```rust
let add = |x: i32, y: i32| x + y;
let mut sum = 0;
[1,2,3].iter().for_each(|&n| sum += n);
```

Capture moves:
```rust
let v = vec![1,2,3];
let f = move || println!("{:?}", v); // moves `v` into closure
f();
```

---

## Smart Pointers & Concurrency Primitives

```rust
use std::rc::Rc;           // single‑thread reference count
use std::sync::{Arc, Mutex, RwLock}; // thread‑safe
use std::cell::RefCell;    // runtime borrow checking (single‑thread)

let shared = Rc::new(RefCell::new(0));
*shared.borrow_mut() += 1;

let shared_mt = Arc::new(Mutex::new(0));
{
    let mut guard = shared_mt.lock().unwrap();
    *guard += 1;
}
```

---

## Threads & Channels

```rust
use std::thread;
use std::sync::mpsc;

let (tx, rx) = mpsc::channel();
let t = thread::spawn(move || { tx.send(42).unwrap(); });
println!("{:?}", rx.recv().unwrap());
t.join().unwrap();
```

Async (runtime required, e.g., Tokio):
```rust
// Cargo.toml: tokio = { version = "1", features = ["full"] }
#[tokio::main]
async fn main() {
    async fn fetch() -> u32 { 7 }
    let x = fetch().await;
    println!("{x}");
}
```

---

## Modules, Crates, `use`

```rust
// lib.rs or main.rs
mod utils;          // looks for utils.rs or utils/mod.rs
pub mod api;        // re-exported

use crate::utils::add;
pub use api::Client;
```

`pub`, `pub(crate)`, `pub(super)`, `pub(in path)` control visibility.

---

## Cargo Quick Commands

```text
cargo new myapp           # binary crate
cargo new --lib mylib     # library crate
cargo build / run / test
cargo fmt / clippy        # format / lints
cargo doc --open
cargo add anyhow          # requires cargo-edit
```

---

## Derive & Common Traits

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
struct Point { x: i32, y: i32 }
```

Useful traits:
- `Display` (user‑facing formatting), `Debug` (developer)
- `From`/`Into`, `AsRef`/`AsMut`
- `Iterator`, `Borrow`, `Deref`

---

## Macros & Attributes

```rust
macro_rules! vec_of_strings {
    ($($s:expr),* $(,)?) => { vec![$($s.to_string()),*] };
}

#[inline]      // hint to inline
#[allow(dead_code)]
#[cfg(test)]
fn helper() {}
```

---

## Formatting & Strings

```rust
println!("{name} = {value:.2}", name="pi", value=3.14159);
let s = format!("{} + {} = {}", 2, 2, 4);
```

---

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    #[should_panic]
    fn panics() { panic!("boom"); }
}
```

Run: `cargo test`

---

## Unsafe (just a peek)

```rust
unsafe fn raw(ptr: *const i32) -> i32 { *ptr }
```

Use sparingly: raw pointers, FFI, manual memory ops.

---

## FFI (C interop)

```rust
#[repr(C)]
pub struct CPoint { x: i32, y: i32 }

extern "C" { fn puts(s: *const i8) -> i32; }
```

---

## Quick Reference Table

- **Ownership**: exactly one owner of `String`, moves on assignment.
- **Borrowing**: `&T` many readers; `&mut T` single writer.
- **Error**: `Result<T, E>`, use `?` to bubble up.
- **Concurrency**: `Arc<T>` share across threads; wrap interior mutability with `Mutex<T>`/`RwLock<T>`.
- **Modules**: `mod name;` + `pub` to expose.
- **Traits**: define behavior; implement with `impl Trait for Type`.
- **Generics**: `fn f<T: Bound>(x: T)` or `where T: Bound`.
- **Lifetimes**: annotate reference relationships when needed.

---

## Formatting Cheats
```rust
// Common std macros
eprintln!("error: {}", "oops");
dbg!(&some_var);            // prints file:line and value
todo!("later");             // panic placeholder
unimplemented!();
```

---

### Tiny Gotchas
- Slicing `String` is by **bytes**, not by Unicode scalars or graphemes.
- Use `&str` for APIs; convert with `impl Into<String>` for ownership.
- Avoid `unwrap()` in libraries—return `Result`.
- Clone only when necessary; prefer borrowing.

---

**Happy hacking!** 🦀
