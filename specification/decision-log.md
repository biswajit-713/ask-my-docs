# Input data
## Choice of books
The Tier 1 set is deliberately chosen so that your chapter detector needs to handle three distinct heading styles:
- BOOK I / BOOK II (Plato, Aristotle, Thucydides)
- SECTION I / SECTION II (Hume)
- Numbered short entries with no headings (Marcus Aurelius)
If your cleaner and chunker handles all three correctly on 5 books, you can be confident it will handle the full 13-book corpus. Start with On Liberty first since it's the fastest — ingest, inspect the chunks, confirm it works, then scale up.