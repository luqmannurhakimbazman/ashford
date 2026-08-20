# Generic syllabus acquisition fixtures

These deterministic fixtures exercise the portable intake path without production course dispatch.

- `two-page-syllabus.pdf` — two-page text layer; SHA-256 `e481c068230a7cb006d783c33b74720f036928317bcb21ecaa9c8392f0084839`.
- `ambiguous-columns.pdf` — deliberately ambiguous column alignment; SHA-256 `32802670de3aa3ef855dda2cfa8d53de2c6701e709d10982cbf1179fdf62b910`.
- `adversarial-syllabus.html` — scripts, styles, templates, comments, links, and subresource URLs that must not execute or fetch; SHA-256 `69c0466256b52f33f8f9f5be2938f56e3d00c96e30cd6c916bd3b8d62dcc948d`.
- `expected-*-extraction.json` — exact `pypdf==6.14.2`, plain-mode, NFC/LF golden output.
- request JSON files — generic proposal/decision examples; event IDs are placeholders replaced by tests or callers.

The unchanged `../st5201x/syllabus2026.pdf` is an additional adversarial input only. It uses the same extractor and has no production adapter, ontology, digest, or course-name dispatch.
