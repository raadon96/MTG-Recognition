---
name: direct-python
description: Direct Python style. Plain data, linear functions, abstractions only when earned, no wasted work. Use when writing, refactoring or reviewing Python, or when another skill needs the directness vocabulary.
---

# Direct Python

Write **direct** code: the reader sees what happens, in order, on plain data. This adapts Casey Muratori's performance-aware, compression-oriented programming to Python. What makes Python code slow (needless work, indirection, scattered logic) is mostly what makes it hard to read, so cutting it improves both. The aim is readability; the performance rules cover only the big pitfalls.

Apply this to the code you write or touch for the task, and leave unrelated code as it is unless asked. An ADR in the repo outranks these rules. If the surrounding code follows a different style, name the conflict in one line and follow the user's choice.

## Vocabulary

Use these terms exactly when naming problems and fixes.

**Indirection**: an extra step between the caller and the work. A wrapper that only forwards (a **pass-through**, after John Ousterhout), Protocol or ABC dispatch, a callable passed in, a registry or factory lookup, a decorator, metaclass or descriptor. Each one hides where control goes next; a direct call shows it.

**Jump**: one time the reader leaves the main path to understand it: into another function, another file, or a dispatch target. Jumps are the unit of readability: count them along the main path, before and after a change. Scattered logic makes jumps without any indirection.

**Plain data**: values with no behaviour attached: `@dataclass(slots=True)`, tuples, dicts, lists, numpy arrays. Functions take plain data and return plain data. Fields are read directly, with no getters or properties for encapsulation's sake.

**Linear function** _(John Carmack)_: a function that reads top to bottom as a sequence of steps, with short comments marking the steps. One longer linear function beats several tiny ones the reader must jump between. Extract a helper when it is called from two places, or when it names a real concept and the call site reads better for it.

**Interface**: everything a caller must know to use the code: the signature, plus its invariants, error modes and ordering constraints.

**Depth** _(David Parnas, John Ousterhout)_: how much behaviour a caller gets per unit of interface they learn. Apply the **deletion test**: imagine deleting the abstraction. If complexity vanishes, it was a pass-through; if it reappears across callers, it was earning its keep. An abstraction with one use is deep only if it hides a decision likely to change: a storage layout, a file format, a library's API. A class around one long step that only one caller uses is not deep; it becomes a linear function. A pass-through is never deep.

**Earned abstraction**: a Protocol, ABC, base class, shared helper, option flag or module that is earned in one of these ways:

- It has two real uses in the code today. A test fake is not a use.
- It is deep.
- It is one of Martin Fowler's two cases where indirection pays ("Inversion of Control Containers and the Dependency Injection pattern", 2004): an implementation chosen per deployment or configuration, and code reused across several applications.

Abstractions are earned through **semantic compression** _(Casey Muratori)_: write the specific code for the case at hand first (usable before reusable); factor out the second time the same logic appears, taking what the real instances share and naming it in the problem's language.

**Multiresolution entry points**: a convenient high-level call plus the lower-level functions it is built from, each callable on its own. Callers and tests drop down a level when they need to, rather than injecting a substitute into the high-level call.

**Locality**: code that works together lives together, in one function or one file, so following a path takes few jumps between files. Split a file when it holds separate concerns that are used independently.

### Problem tags

Tag every problem you report with one of:

- **Unearned abstraction**: fails the earned-abstraction test.
- **Indirection**: indirection where a direct call would do.
- **Scattered logic**: one sequence of steps spread across tiny functions or files.

## Rules

### Kinds are a field, not subclasses

Kinds with the same fields are one dataclass with a `kind` field, plus a table indexed by kind when only constants differ, or `match` on the kind when the steps differ. Kinds whose fields differ are one dataclass per kind, joined by a union type (`Square | Circle`) and handled with `match` on the class. Either way the behaviour lives in one function, so every case is visible in one place, and no subclass holds it.

Avoid:

```python
class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

class Square(Shape):
    def __init__(self, side): self.side = side
    def area(self): return self.side * self.side

class Circle(Shape):
    def __init__(self, r): self.r = r
    def area(self): return math.pi * self.r * self.r

total = sum(s.area() for s in shapes)
```

Prefer, when the kinds share their fields:

```python
AREA_COEFF = {"rectangle": 1.0, "triangle": 0.5}

@dataclass(slots=True)
class Polygon:
    kind: Literal["rectangle", "triangle"]
    width: float
    height: float

def total_area(polygons: list[Polygon]) -> float:
    return sum(AREA_COEFF[p.kind] * p.width * p.height for p in polygons)
```

Prefer, when their fields differ:

```python
@dataclass(slots=True)
class Square:
    side: float

@dataclass(slots=True)
class Circle:
    radius: float

def area(shape: Square | Circle) -> float:
    match shape:
        case Square(side=side):
            return side * side
        case Circle(radius=radius):
            return math.pi * radius * radius
```

### Test seams follow the dependency order

A test seam is a function that already exists, called by a test with plain data. When logic depends on something heavy (an ML model, the network, a database, the filesystem), the **dependency order** _(Mark Seemann, "Dependency rejection" and "Impureim sandwich"; Gary Bernhardt, "Boundaries"; Brandon Rhodes, "Hoisting Your I/O", 2015)_ decides how a test reaches it. Take the first step that applies:

1. **Pass the value.** Move the I/O up to the caller, so the logic takes plain data. Tests call the logic with data; the caller is the I/O plus one call, covered by regression tests on real inputs.
2. **Otherwise, for a dependency that leaves the process (the network, a database), pass it through an argument the code already takes for it** (an HTTP client). A test passes a fake through that argument.
3. **Add a Protocol or injection point only when two production implementations exist today.**

Testability alone never earns a new Protocol, callable parameter or injection point.

Avoid (an injection point added for the test):

```python
class TextRecognizer(Protocol):
    def recognize(self, image: np.ndarray) -> list[str]: ...

class ReceiptReader:
    def __init__(self, recognizer: TextRecognizer | None = None):
        self._recognizer = recognizer or TesseractRecognizer()

    def read(self, image: np.ndarray) -> Receipt:
        lines = self._recognizer.recognize(image)
        return Receipt(total=_find_total(lines), date=_find_date(lines))

# test
reader = ReceiptReader(recognizer=FakeRecognizer(["TOTAL 12.50"]))
assert reader.read(blank_image).total == 12.50
```

Prefer (step 1):

```python
def read_receipt(image: np.ndarray) -> Receipt:
    lines = pytesseract.image_to_string(image).splitlines()
    return parse_receipt(lines)

def parse_receipt(lines: list[str]) -> Receipt:
    total, date = None, None
    for line in lines:
        # total: "TOTAL 12.50"
        if line.startswith("TOTAL"):
            total = float(line.split()[-1])
        # date: "2026-10-21"
        elif DATE_RE.fullmatch(line):
            date = line
    return Receipt(total=total, date=date)

# test: the lower-level entry point, on plain data
assert parse_receipt(["TOTAL 12.50"]).total == 12.50
```

`read_receipt` is covered by a regression test on real receipt images.

### Callers above helpers

A function sits below its callers; a helper with several callers goes below the first one. The reader meets each step before its details, as in a linear function. This covers the functions a change writes or touches; the rest of the file keeps its order.

In the test-seam example above, `read_receipt` sits above `parse_receipt`, the helper it calls.

### Comments say why

Names and structure say what the code does. A comment says what they can't: the reason for a heuristic, a threshold or a workaround. Short step comments in a linear function stay. A deep abstraction may also have a docstring stating its interface: what callers can rely on without reading the body. This covers the comments a change writes or touches.

Avoid:

```python
# divide the matched count by the number of words
overlap = matched / len(words)
```

Prefer:

```python
# long names match more words by chance, so normalise by length
overlap = matched / len(words)
```

### Big performance pitfalls

Python runs many machine instructions per line, so the large wins come from removing work. When the data is large enough to matter:

- Batch I/O: one query, request or file write for many items, outside the loop.
- Decode or parse each input once and pass the result along.
- Use a set or dict for membership and lookup.
- Use one numpy expression for math over large numeric arrays.
- Build invariant structures once, before the loop.

A claim that one version is faster comes with a measurement: `timeit`, or `time.perf_counter` over repeated runs, taking the minimum.

## Before finishing

- Every Protocol, ABC, base class, wrapper, helper and file you added is an earned abstraction, or a helper that names a real concept.
- The main path has as few jumps as a linear function allows.
- Each function you wrote or touched sits below its first caller.
- Each comment you wrote or touched says why, marks a step, or is the interface docstring of a deep abstraction.
- Logic that depends on something heavy follows the dependency order, and tests call existing entry points with plain data.
- Large data avoids all five pitfalls.
- The code has the features and parameters today's requirements need, and no more. Its interface may be named in the problem's terms rather than its one caller's.

## Rejected framings

- **`match` is complex, protocols are simple** (Rich Hickey, "Simple Made Easy"): here a protocol is an abstraction like any other and has to be earned.
- **Subclass polymorphism instead of a `kind` field** (Hynek Schlawack, "Subclassing in Python Redux"): his objection is to one class with optional fields that depend on the kind, which the union of dataclasses answers; his class per kind is kept, without behaviour.
- **Rule of Three**: factor out on the second copy, after Muratori.
- **"Somewhat general-purpose" interfaces** (Ousterhout): naming the interface in the problem's terms is kept; extra features or parameters are not.
- **Depth as a ratio of implementation lines to interface lines** (Ousterhout): it rewards padding the implementation.
- **Very short functions** (Robert C. Martin, in the `aposd-vs-clean-code` debate with Ousterhout, 2024–2025): linear functions instead.
- **Explicit dependencies added for easier tests** (Harry Percival and Bob Gregory, *Architecture Patterns with Python*, ch. 13): step 1 of the dependency order removes the need.

People editing this skill: the sources behind it are checked in https://github.com/decksray/MTG-Recognition/issues/106.
