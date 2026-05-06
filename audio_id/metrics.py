"""Prometheus counters and histograms for the Audio ID system."""

from prometheus_client import Counter, Gauge, Histogram

QUERY_LATENCY = Histogram(
    "query_latency_seconds",
    "End-to-end query latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5],
)

QUERIES_TOTAL = Counter(
    "queries_total",
    "Total queries received",
)

MATCH_TYPE = Counter(
    "match_type_total",
    "Matches by type",
    ["type"],
)

INDEX_SONGS = Gauge(
    "index_songs_total",
    "Songs in index",
)

INDEX_POSTINGS = Gauge(
    "index_hash_postings_total",
    "Hash postings in index",
)
