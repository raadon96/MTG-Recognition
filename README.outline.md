# PROTOTYPE: README outline (throwaway, for "Draft the README outline", #257)

Rough outline of the users-only README. Order is the proposal; wording is a first take to react to.
`TODO(measure)` marks a number taken from the published image, not guessed.

---

# MTG Recognition

**Opening paragraph** (what / benefit / difference, no jargon):

> Take a photo of your Magic: The Gathering cards and find each card's exact printing: the set, collector number and art, not just the name. This repo is a prototype of the recognition engine, with a simple web app on top to try it: it runs on your own computer from one Docker command and handles several cards in one photo.

(No mention of future plans such as a WASM port; this paragraph is rewritten if the repo becomes a product.)

**Demo GIF** (`docs/assets/demo.gif`): directly under the paragraph. Content decided in "Decide what the demo GIF shows" (#262).

**Facts** (bold keyword, then evidence):

- **Every printing.** Knows all 115,000+ printings, double-faced cards included.
- **Several cards per photo.** Finds and straightens each card in a photo or scan.
- **Runs locally.** Your photos are recognised on your computer and never uploaded; only the card pictures in the results are loaded from Scryfall.
- **One command.** No install beyond Docker; runs on `linux/amd64` and `linux/arm64`.

---

## Run it

1. Install Docker: Docker Desktop on Windows and macOS, Docker Engine on Linux (one link each).
2. Run:

   ```bash
   docker run --rm -p 8501:8501 ghcr.io/raadon96/mtg-recognition:latest
   ```

3. Open <http://localhost:8501>.

Notes under the steps, one line each:

- The first run downloads the image, about TODO(measure) compressed.
- **Fixed version**: replace `latest` with a release number such as `0.10.0` (link to Releases).
- **Stop it**: `Ctrl+C` in the terminal. `--rm` removes the container; the image stays for next time.
- **Use more CPU threads**: add `-e THREAD_COUNT=8` (default 4).

## New sets

Each release knows the sets out when it was built; cards from newer sets aren't recognised. A later release adds them (see Releases); to get it, pull the image again:

```bash
docker pull ghcr.io/raadon96/mtg-recognition:latest
```

No promise of when releases come out.

## Contributing

One line: to work on the code, run the benchmark or build the image yourself, see CONTRIBUTING.md.

## License

- AGPL-3.0-only (LICENSE). Model licences in THIRD_PARTY_NOTICES.md.
- Credits: card data from MTGJSON, card images from Scryfall.
- Fan Content Policy notice, word for word, in a blockquote.

---

## Dropped from today's README

- The Compose quick start, clone steps, `docker compose logs/down -v` and the "models download on first start" note: contributors use Compose, so it moves to CONTRIBUTING.md; the model note is wrong since the image holds the weights.
- "Setup for Development" and every `mtg reference ...` paragraph: CONTRIBUTING.md and the commands' `--help`.
- The "Decksray" prefix in the title.
- Benchmark figures: users can't see the benchmark images, so the numbers tell them nothing.
- No badges, no table of contents (under two screens), no CLI, no "Use it" section (the GIF shows it), no "Remove it" section.
