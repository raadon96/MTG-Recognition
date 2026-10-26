---
name: direct-python
description: Direct Python style. Plain data, linear functions, abstractions only when earned, no wasted work. Use when writing, refactoring or reviewing Python, or when another skill needs the directness vocabulary.
---

# Direct Python

Write **direct** code: the reader sees what happens, in order, on plain data. This adapts Casey Muratori's performance-aware, compression-oriented programming to Python. What makes Python code slow (needless work, indirection, scattered logic) is mostly what makes it hard to read, so cutting it improves both. The aim is readability; the performance rules cover only the big pitfalls.

Apply this to the code you write or touch for the task, and leave unrelated code as it is unless asked. An ADR in the repo outranks these rules. If the surrounding code follows a different style, name the conflict in one line and follow the user's choice.

## Vocabulary

Use these terms exactly when naming problems and fixes.

**Indirection**: an extra step between the caller and the work. A wrapper that only forwards (a **pass-through**), Protocol or ABC dispatch, a callable passed in, a registry or factory lookup, a decorator, metaclass or descriptor. Each one hides where control goes next; a direct call shows it.

**Jump**: one time the reader leaves the main path to understand it: into another function, another file, or a dispatch target. Jumps are the unit of readability: count them along the main path, before and after a change. Scattered logic makes jumps without any indirection.

**Plain data**: values with no behaviour attached: `@dataclass(slots=True)`, tuples, dicts, lists, numpy arrays. Functions take plain data and return plain data. Fields are read directly, with no getters or properties for encapsulation's sake. Kinds that behave differently are a `kind` field plus `match`, or a table indexed by kind when only constants differ, so every case is visible in one place.

**Linear function**: a function that reads top to bottom as a sequence of steps, with short comments marking the steps. One longer linear function beats several tiny ones the reader must jump between. Extract a helper when it is called from two places, or when it names a real concept and the call site reads better for it.

**Earned abstraction**: a Protocol, ABC, base class, shared helper, option flag or module with at least two real uses in the code today. A test fake is not a use. Abstractions are earned through **semantic compression**: write the specific code for the case at hand first (usable before reusable); when the same logic appears a second time, factor out what the real instances share and name it in the problem's language.

**Multiresolution entry points**: a convenient high-level call plus the lower-level functions it is built from, each callable on its own. Callers and tests drop down a level when they need to, rather than injecting a substitute into the high-level call.

**Locality**: code that works together lives together, in one function or one file, so following a path takes few jumps between files. Split a file when it holds separate concerns that are used independently.

### Problem tags

Tag every problem you report with one of:

- **Unearned abstraction**: fails the earned-abstraction test.
- **Indirection**: indirection where a direct call would do.
- **Scattered logic**: one sequence of steps spread across tiny functions or files.

## Rules

### Dependencies are plain arguments

Pass what a function needs as ordinary parameters: values, data, or the object it works on. Call real functions directly. A dependency-injection framework, a service locator, or a parameter that exists only so something can be substituted is indirection.

### Test seams are existing entry points

A test seam is a function that already exists, called by a test with plain data. Testability alone never earns a new Protocol, callable parameter or injection point. To test logic without a heavy dependency (an ML model, the network, a database), let the logic run on the plain data the heavy step produces, expose it as a lower-level entry point, and test that directly. Cover the heavy step itself with regression tests on real inputs.

Avoid:

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

Prefer:

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

### Functions read top to bottom

A function sits below its callers; a helper with several callers goes below the first one. The reader meets each step before its details, as in a linear function. This covers the functions a change writes or touches, not a reorder of the rest of the file.

```python
def read_receipt(image: np.ndarray) -> Receipt:
    lines = pytesseract.image_to_string(image).splitlines()
    return parse_receipt(lines)

def parse_receipt(lines: list[str]) -> Receipt: ...
```

### Comments say why

Names and structure say what the code does. A comment says what they can't: the reason for a heuristic, a threshold or a workaround. Short step comments in a linear function stay. This covers the comments a change writes or touches.

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

## Examples

### Plain data over a type hierarchy

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

Prefer:

```python
AREA_COEFF = {"square": 1.0, "rectangle": 1.0, "triangle": 0.5, "circle": math.pi}

@dataclass(slots=True)
class Shape:
    kind: str
    width: float
    height: float  # squares and circles store width in height too

def total_area(shapes: list[Shape]) -> float:
    return sum(AREA_COEFF[s.kind] * s.width * s.height for s in shapes)
```

## Before finishing

- Every Protocol, ABC, base class, wrapper, helper and file you added is an earned abstraction, or a helper that names a real concept.
- The main path has as few jumps as a linear function allows.
- Each function you added sits below its first caller.
- Each comment you wrote or touched says why, or marks a step.
- Tests call existing entry points with plain data.
- Large data avoids all five pitfalls.
- The code is as general as today's requirements, and no more.
