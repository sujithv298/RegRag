"""Tests for `regrag.pii.scrubber.RegexPIIScrubber`.

We don't test PresidioPIIScrubber here — it requires loading spaCy models
that need network access. That belongs in tests/integration/.
"""

from __future__ import annotations

from regrag.pii import PIIScrubber, RegexPIIScrubber


def test_implements_protocol() -> None:
    assert isinstance(RegexPIIScrubber(), PIIScrubber)


def test_scrubs_ssn() -> None:
    out = RegexPIIScrubber().scrub("My SSN is 123-45-6789, can you help?")
    assert "123-45-6789" not in out.text
    assert "[REDACTED:SSN]" in out.text
    assert out.redaction_count == 1


def test_scrubs_email() -> None:
    out = RegexPIIScrubber().scrub("Send the response to user@bank.example.com please.")
    assert "user@bank.example.com" not in out.text
    assert "[REDACTED:EMAIL]" in out.text
    assert out.redaction_count == 1


def test_scrubs_phone() -> None:
    out = RegexPIIScrubber().scrub("Call me at 555-123-4567 about the dispute.")
    assert "555-123-4567" not in out.text
    assert "[REDACTED:PHONE]" in out.text


def test_scrubs_credit_card() -> None:
    out = RegexPIIScrubber().scrub("Card number 4111 1111 1111 1111 was used.")
    assert "4111 1111 1111 1111" not in out.text
    assert "[REDACTED:CARD]" in out.text


def test_scrubs_multiple_in_one_message() -> None:
    out = RegexPIIScrubber().scrub(
        "I'm 123-45-6789 and my email is me@example.com, call 555-123-4567."
    )
    assert out.redaction_count == 3


def test_no_pii_no_redaction() -> None:
    out = RegexPIIScrubber().scrub("What does Reg E say about consumer liability?")
    assert out.redaction_count == 0
    assert out.text == "What does Reg E say about consumer liability?"


def test_does_not_alter_chunk_id_style_strings() -> None:
    """Citation references like '1005.6(b)(1)' shouldn't be confused with
    PII. SSN regex is specific (3-2-4 digit groups with hyphens) so it
    shouldn't match '1005.6(b)(1)'."""
    text = "Per 12 CFR 1005.6(b)(1), the consumer's liability is capped."
    out = RegexPIIScrubber().scrub(text)
    assert out.redaction_count == 0
    assert out.text == text
