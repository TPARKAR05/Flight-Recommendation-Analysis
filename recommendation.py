"""
Recommendation Engine
---------------------
Ranks flights using weighted min-max normalisation.

Since Aviationstack free plan does not provide price data, ranking is
based on duration and delay only:
    Score = 0.6 * norm(duration) + 0.4 * norm(delay)
Lower score = better flight.
"""


def _norm(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def recommend_flights(
    flights: list[dict],
    w_duration: float = 0.5,
    w_delay:    float = 0.3,
    w_price:    float = 0.2,
) -> list[dict]:
    if not flights:
        return []

    nd = _norm([f["duration_minutes"] for f in flights])
    nl = _norm([f["delay_minutes"]    for f in flights])
    np_ = _norm([f["price"]           for f in flights])

    for i, f in enumerate(flights):
        f["score"] = round(w_duration * nd[i] + w_delay * nl[i] + w_price * np_[i], 4)

    flights.sort(key=lambda x: x["score"])
    for rank, f in enumerate(flights, 1):
        f["rank"] = rank

    return flights