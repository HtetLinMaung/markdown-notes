# Go (Golang) Syntax Cheat Sheet

> Quick, compact, and copy-paste friendly. Use it like duct tape for
> your brain.

------------------------------------------------------------------------

## Getting Started

``` bash
go version
go mod init example.com/awesome   # start a module
go run .                          # run main package
go build                          # build binary
go test ./...                     # run tests
go fmt ./...                      # format
```

------------------------------------------------------------------------

## Basics

``` go
package main

import (
    "fmt"
    "errors"
)

func main() {
    fmt.Println("Hello, Go!")
}
```

### Variables, Constants, Types

``` go
var x int = 42
y := 3.14            // short declare (inside funcs)
const Pi = 3.14159

type ID int
var userID ID = 7

// Zero values: 0, 0.0, "", false, nil
```

### Built-in Types

-   Numeric: `int`, `int8/16/32/64`, `uint*`, `float32/64`,
    `complex64/128`
-   Text: `string`, `rune` (int32), `byte` (uint8)
-   Bool: `bool`
-   Aggregates: `array`, `slice`, `map`, `struct`
-   Pointers: `*T` (no pointer arithmetic)

------------------------------------------------------------------------

## Functions

``` go
func add(a, b int) int { return a + b }

func split(sum int) (x, y int) {  // named returns
    x = sum * 4 / 9
    y = sum - x
    return
}

func must(ok bool) error {
    if !ok { return errors.New("nope") }
    return nil
}

// Variadic
func sum(nums ...int) int {
    total := 0
    for _, n := range nums { total += n }
    return total
}
```

------------------------------------------------------------------------

## Control Flow

``` go
if n := len(s); n > 0 { /* use n */ } else { /* ... */ }

for i := 0; i < 10; i++ { /* ... */ }         // classic
for i < 10 { /* ... */ }                      // while
for { break }                                 // infinite

switch v := anyVal.(type) {
case int:    fmt.Println("int", v)
case string: fmt.Println("string", v)
default:     fmt.Println("unknown")
}

// Defer, Panic, Recover
func safe(fn func()) (err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic: %v", r)
        }
    }()
    fn()
    return
}
```

------------------------------------------------------------------------

## Arrays, Slices, Maps

``` go
// Arrays (fixed size)
var a [3]int = [3]int{1,2,3}

// Slices (dynamic views over arrays)
s := []int{1,2,3}
s = append(s, 4, 5)
t := make([]int, 0, 10)   // len 0, cap 10
u := s[1:3]               // slicing (half-open)

// Copy
dst := make([]int, len(s))
copy(dst, s)

// Map
m := map[string]int{"alice": 1}
m["bob"] = 2
v, ok := m["carol"]       // ok idiom
delete(m, "alice")

// Note: writing to a nil map panics; use make first.
```

------------------------------------------------------------------------

## Structs, Methods, Embedding

``` go
type Point struct{ X, Y float64 }

func (p Point) Len() float64 { return p.X*p.X + p.Y*p.Y }
func (p *Point) Move(dx, dy float64) { p.X += dx; p.Y += dy }

type ColoredPoint struct {
    Point            // embedding (promotes methods/fields)
    Color string
}
```

------------------------------------------------------------------------

## Interfaces

``` go
type Stringer interface {
    String() string
}

type User struct{ Name string }
func (u User) String() string { return "User(" + u.Name + ")" }

func Print(s fmt.Stringer) { fmt.Println(s.String()) }
// Interface satisfaction is implicit.
```

------------------------------------------------------------------------

## Generics (Go 1.18+)

``` go
// Constraints
type Number interface {
    ~int | ~int64 | ~float64 // ~ allows type aliases
}

func Max[T Number](a, b T) T {
    if a > b { return a }
    return b
}

// Generic types
type Pair[T any] struct{ A, B T }
```

------------------------------------------------------------------------

## Pointers

``` go
p := Point{3,4}
pp := &p
pp.Move(1, 1)      // method works on pointer receiver
fmt.Println(p.X)   // 4
```

------------------------------------------------------------------------

## Concurrency

``` go
// Goroutines
go work()

// Channels
ch := make(chan int)
go func() {
    defer close(ch)
    for i := 0; i < 3; i++ { ch <- i }
}()

for v := range ch { fmt.Println(v) }  // 0,1,2

// Buffered channels
bch := make(chan string, 2)
bch <- "hi"; bch <- "there"

// Select
select {
case v := <-ch:
    fmt.Println("got", v)
case <-time.After(100 * time.Millisecond):
    fmt.Println("timeout")
}
```

### Context (cancellation & deadlines)

``` go
ctx, cancel := context.WithTimeout(context.Background(), time.Second)
defer cancel()

select {
case <-ctx.Done():
    fmt.Println("done:", ctx.Err())
}
```

------------------------------------------------------------------------

## Errors

``` go
f, err := os.Open("file.txt")
if err != nil {
    return fmt.Errorf("open: %w", err)  // wrap
}
defer f.Close()
```

------------------------------------------------------------------------

## Packages & Modules

    project/
      go.mod
      go.sum
      cmd/app/main.go
      internal/pkg/thing.go

``` go
// Exported identifiers start with Capital letters.
package mathx

func Add(a, b int) int { return a + b }
```

------------------------------------------------------------------------

## I/O, JSON, HTTP (Tiny Samplers)

``` go
// Read file
data, err := os.ReadFile("in.txt")

// JSON
type Person struct {
    Name string `json:"name"`
    Age  int    `json:"age"`
}
var p Person
_ = json.Unmarshal(data, &p)
b, _ := json.MarshalIndent(p, "", "  ")

// HTTP server
http.HandleFunc("/hello", func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintln(w, "hello")
})
log.Fatal(http.ListenAndServe(":8080", nil))

// HTTP client
resp, err := http.Get("https://example.com")
defer resp.Body.Close()
body, _ := io.ReadAll(resp.Body)
```

------------------------------------------------------------------------

## Testing

``` go
// file: add_test.go
package mathx_test

import (
    "testing"
    "example.com/awesome/mathx"
)

func TestAdd(t *testing.T) {
    if got := mathx.Add(2, 2); got != 4 {
        t.Fatalf("want 4, got %d", got)
    }
}
```

Run: `go test -v ./...`

------------------------------------------------------------------------

## Common Gotchas

-   `for range` on map gives random order (by design).
-   Don't take address of loop variable when capturing in
    goroutines---copy it first.
-   `nil` slice is OK to `append` to; `nil` map is **not** OK to write
    to.
-   Use `defer` to close resources *as soon as you open them*.
-   Prefer errors over panics in normal flow.

------------------------------------------------------------------------

## Mini Patterns

``` go
// Guard clause
if err := do(); err != nil { return err }

// Timeout wrapper
func withTimeout(d time.Duration, f func(ctx context.Context) error) error {
    ctx, cancel := context.WithTimeout(context.Background(), d)
    defer cancel()
    return f(ctx)
}
```

------------------------------------------------------------------------

## Formatting & Lint

-   Always run `go fmt` (or `gofmt -s`).
-   Consider `go vet` and a linter (e.g., `golangci-lint`) for extra
    checks.

------------------------------------------------------------------------
