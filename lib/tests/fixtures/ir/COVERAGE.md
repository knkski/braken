# Shared derivation IR corpus coverage

This generated report counts grammars that exercise each IR feature. It is a breadth
signal, not a substitute for the exact curated goldens in [`GALLERY.md`](GALLERY.md).

| Corpus | Grammars | Parameters | Context | Structural context | Filters | Dynamic conditions | Dynamic weights | Weighted | Branches | Builtins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Curated IR | 14 | 6 | 6 | 1 | 5 | 4 | 1 | 2 | 4 | 2 |
| ABOP pass | 94 | 35 | 15 | 0 | 10 | 22 | 0 | 3 | 56 | 2 |
| Preset pass | 11 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| Web pass | 1011 | 304 | 6 | 2 | 5 | 238 | 0 | 18 | 389 | 1 |
