"""English stopword list, conservatively trimmed for legal text.

We use this **only at query time** — the index keeps every token so that
citation strings and quoted phrases remain searchable in full. Stripping
common query words (what, does, the, an) before scoring prevents the BM25
"the rarest matched token wins" failure mode where an accidental match on
"does" outranks a deliberate match on "1005.6".

Things deliberately *kept* even though stock NLP stopword lists drop them:

  - Modal verbs (`shall`, `must`, `may`, `should`, `would`, `can`) — these
    are load-bearing in regulatory text. "The institution **shall** not"
    means something different from "the institution not."
  - Quantifiers (`all`, `any`, `every`, `each`) — also legally meaningful.
  - Negation (`not`, `no`, `nor`) — search isn't logic, but these tend to
    co-occur with the actual topic words and don't usually create false
    positives.

If you fork this template for a non-legal corpus, you may want to use
NLTK's stopword list instead.
"""

from __future__ import annotations

ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        # Articles
        "a",
        "an",
        "the",
        # Question words (queries are full of these; content rarely is)
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        # Auxiliary "do/be/have"
        "do",
        "does",
        "did",
        "doing",
        "done",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        # Pronouns
        "i",
        "me",
        "my",
        "mine",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "we",
        "us",
        "our",
        "ours",
        "they",
        "them",
        "their",
        "theirs",
        "this",
        "that",
        "these",
        "those",
        # Conjunctions
        "and",
        "or",
        "but",
        # Most prepositions
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "to",
        "as",
        "about",
        "into",
        "onto",
        # Common query verbs that don't carry topical meaning
        "say",
        "says",
        "said",
        "tell",
        "tells",
        "told",
        "mean",
        "means",
        "meant",
        # Fillers
        "also",
        "just",
    }
)
