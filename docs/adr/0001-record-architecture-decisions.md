# ADR 0001: Record architecture decisions

**Status:** Accepted
**Date:** 2026-05-01

## Context

This project will accumulate non-obvious design decisions: why Chroma over pgvector by default, why hybrid retrieval over pure-dense, why fail-closed citation verification, why deterministic over agentic in v1. A forker reading the source six months from now will ask "why?" of every one of these.

## Decision

We use [Architecture Decision Records](https://adr.github.io/) in `docs/adr/`. One file per decision. Numbered sequentially. Format: Context / Decision / Consequences. Status of `Proposed`, `Accepted`, `Deprecated`, or `Superseded by ADR-NNNN`.

## Consequences

- A reviewer can see why a decision was made without git-archaeology through PR descriptions.
- Forkers who want to make a different choice (e.g., default to pgvector) know what they're trading off.
- Decisions that get overturned are visible: the old ADR stays, marked `Superseded`.
