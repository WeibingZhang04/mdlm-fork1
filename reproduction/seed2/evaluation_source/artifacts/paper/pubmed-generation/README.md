# PubMed generation comparison

This directory contains the fail-closed, source-bound analysis for the pinned
Scientific Papers PubMed generation panel. The queue completed 32 shards (16
per adapter arm), covering 256 prompts and 1,024 paired sample draws at each
declared comparison. `bundle.json` atomically binds the queue-completion
evidence, verified unions, and 20,000-resample paired comparison.

The immutable controller checkout was
`446cb8abe1a19db74f16efa658d17882812a0e76`; the immutable generation runner
was `09f89c00bbf8c65f679cd40b92609754608817b8`. The reviewed WikiText launch
gate SHA-256 was
`a33ae2b0ea1834ced45047fe50ecffaef69b9571e9cd67199054edbe02b7ee86`.
The harvested bundle SHA-256 is
`9cd8f4b26625c390e1aac953a070dbc83506336915d64c1e7ac02012f8a8f46e`.

Endpoint differences in `paired-adapter-comparison.json` are treatment minus
baseline. Thus, a positive reference-LM NLL difference is worse and a negative
repetition-rate difference is better.
