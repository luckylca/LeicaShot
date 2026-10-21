# ADR 0001: Independent implementation and reference boundary

- Status: accepted
- Date: 2026-10-21

## Decision

Laika is a clean, independent Python project with its own package, API, history, and release version. The Android reference repository is kept outside the Laika worktree and is used to understand model contracts and algorithm behavior. No reference source files, test fixtures, model weights, or history are imported into Laika.

## Reasons

The reference project identifies its original source and bundled M9 model as personal non-commercial material. Some other checkpoints have model-card licenses that may not resolve dataset or checkpoint redistribution rights. Laika must preserve that distinction and must not infer permission from public availability or from an upstream code license alone.

## Consequences

Laika's project-source license remains undecided in this bootstrap. A model cannot enter the default distribution until its exact checkpoint and intended use have documented permissions. The first model integration should use an independently reviewed checkpoint or remain opt-in while permissions are unresolved.
