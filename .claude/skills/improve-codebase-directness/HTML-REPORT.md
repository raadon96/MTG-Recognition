# HTML Report Format

The directness review is a single self-contained HTML file. Tailwind and Mermaid come from CDNs. Mermaid handles call flows; hand-built divs and inline SVG handle the editorial visuals (file maps, code outlines). Mix the two so the diagrams don't all look the same.

## Scaffold

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Directness review for {{repo name}}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
      mermaid.initialize({ startOnLoad: true, theme: "neutral", securityLevel: "loose" });
    </script>
    <style>
      .abstraction { border-style: dashed; }
      .indirection { stroke: #dc2626; }
      .jump { font-variant-numeric: tabular-nums; }
    </style>
  </head>
  <body class="bg-stone-50 text-slate-900 font-sans">
    <main class="max-w-5xl mx-auto px-6 py-12 space-y-12">
      <header>...</header>
      <section id="candidates" class="space-y-10">...</section>
      <section id="top-recommendation">...</section>
    </main>
  </body>
</html>
```

## Header

Repo name, date, and a compact legend: solid box = function, dashed box = abstraction (Protocol, ABC, wrapper), red arrow = indirection, numbered circle = jump. Straight into the candidates.

## Candidate card

The diagrams carry the weight. Prose is sparse and uses the `direct-python` vocabulary.

Each candidate is one `<article>`:

- **Title**: short, names the change (e.g. "Read the receipt in one linear function").
- **Badge row**: recommendation strength (`Strong` = emerald, `Worth exploring` = amber, `Speculative` = slate), plus the problem tag (`unearned abstraction`, `indirection`, `scattered logic`).
- **Files**: monospaced list, `font-mono text-sm`.
- **Before / After diagram**: the centrepiece, two columns side by side. See the patterns below.
- **Jumps**: one line, `before 7 → after 2`, on the main path the diagram shows.
- **Problem**: one sentence. What the reader has to jump through.
- **Solution**: one sentence. What disappears or moves.
- **Wins**: bullets, ≤6 words each, e.g. "Delete 3 pass-throughs", "Main path fits one screen", "Tests call `parse_receipt` directly".
- **Tests**: the entry points tests call afterwards.
- **ADR callout** (if applicable): one line in an amber-tinted box.

If the diagram needs a paragraph to be understood, redraw the diagram.

## Diagram patterns

Pick the pattern that fits the candidate.

### Call flow (Mermaid, the workhorse)

A `flowchart` of the main path with each jump numbered. Before: the hops through wrappers and Protocols, indirection edges in red, abstractions dashed. After: the same path with the hops gone.

```html
<div class="rounded-lg border border-slate-200 bg-white p-4">
  <pre class="mermaid">
    flowchart LR
      A[read_receipt] -->|1| B[ReceiptReader.read]
      B -->|2| C{{TextRecognizer}}
      C -->|3| D[TesseractRecognizer]
      classDef abstraction stroke-dasharray:4 4;
      class C abstraction
      linkStyle 1,2 stroke:#dc2626,stroke-width:2px;
  </pre>
</div>
```

### Call-graph collapse

Before: a tree of one-use helpers as nested boxes. After: one linear function, with the former helpers shown as commented steps inside it.

### File map (locality)

One box per file, arrows for each time the main path crosses into another file. Before: the path zigzags across files. After: it stays in one or two.

### Code outline

Side-by-side outlines of the code's shape: function names with their line counts as bar widths. Before: many thin bars. After: one long bar plus the entry points tests call.

## Style

- Editorial, not dashboard. Generous whitespace; `font-serif` headings work well with stone/slate.
- One accent (emerald or indigo), red for indirection, amber for warnings.
- Keep diagrams ~320px tall so before and after sit side by side.
- Labels inside diagrams in `text-xs uppercase tracking-wider`.
- The only scripts are the Tailwind CDN and the Mermaid ESM import.

## Top recommendation section

One larger card: candidate name, one sentence on why, anchor link to its card.

## Tone

Plain and concise. **Use exactly:** indirection, jump, plain data, linear function, earned abstraction, multiresolution entry points, locality, and the three problem tags. Wins name the gain in those terms or as a count (jumps, files, lines, abstractions removed), e.g. _"locality: the main path stays in one file"_, _"plain data: three subclasses become a table"_.

If a sentence could be a bullet, make it a bullet. If a bullet could be cut, cut it.
