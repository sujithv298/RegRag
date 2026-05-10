"""Audit log: examiner-grade record of every query.

Goal: an examiner can pick any historical query and reconstruct exactly
what the system saw, what it retrieved, what it generated, and why it
answered or refused — months after the fact.

See `docs/audit-log-schema.md` for the field-by-field contract.
"""

from __future__ import annotations

from regrag.audit.logger import AuditLogger, make_record
from regrag.audit.schema import (
    SCHEMA_VERSION,
    AuditCitationRecord,
    AuditRecord,
    Outcome,
)

__all__ = [
    "SCHEMA_VERSION",
    "AuditCitationRecord",
    "AuditLogger",
    "AuditRecord",
    "Outcome",
    "make_record",
]
