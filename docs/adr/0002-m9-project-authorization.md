# ADR 0002: Develop with authorized model use and local assets

**Status:** Accepted
**Date:** 2026-10-22

## Context

The reference implementation identifies M9 series 3.0 epoch11 as its core AI
color model. Laika's initial permission review deferred the model while treating
the reference repository's personal non-commercial license as the only
available authorization information.

The user subsequently confirmed that the required authorization for Laika to
use M9 and the other planned models is in place. This is the project decision
for development and use. It supersedes earlier M9 or other model-use exclusions
without rewriting or deleting prior Git history.

## Decision

- M9 remains Laika's core AI color predictor; no substitute color model is
  planned. The other required models remain in the development plan.
- The exact `leica-m9-v3-epoch11.fp16.onnx` contract and SHA-256 are recorded in
  [`models/m9-local.json`](../../models/m9-local.json).
- The model is supplied through a local path and is verified before ONNX
  Runtime creates a session.
- The model weight is not committed to Git or published by Laika. User
  authorization for project use does not by itself establish public
  redistribution rights.
- The independent Python adapter and Lab fusion are tested separately and as a
  full M9 photo pipeline.

## Consequences

CPU inference, Python SDK, and CLI development can proceed with the actual M9
model. CI remains model-free; local or controlled model tests require a locally
provided file whose SHA-256 matches the manifest. The broader pipeline still
needs measured memory and quality results before any 4 GiB or 2 GiB claim.
