"""Render an airline `info` dict as free text, so the extraction arm has something
unstructured to read.

The point of the experiment this feeds is that the main study hands BOTH branches a
structured dict, which is not what a deployment gets. Here the case arrives as prose
and the pipeline has to recover the dict before the program can run.

Deterministic given (info, seed): same case, same text, every run. That matters
because the call cache is keyed on the prompt -- a renderer that drifted would
silently re-buy every call and make the run unreproducible.

What the prose does NOT do is add facts the ruleset acts on. Every distractor here
(confirmation code, seat, departure time, meal note) is inert with respect to the
reference computation, verified by tools/run_extract.py's inertness check. A
distractor that changed the answer would make the arms incomparable rather than
harder.

Honest limit: this is templated prose with randomized surface form, not scraped
human writing. It varies phrasing, clause order, units, and which facts are stated
indirectly; it does not contain genuine ambiguity, contradiction, or missing data.
Read the resulting numbers as the EASY end of unstructured input.
"""
from __future__ import annotations

import random

# routine label -> cities that sit in it. The label is what the ruleset keys on, so
# the prose names a city and the extractor has to map it back. This is the one piece
# of world knowledge the extraction step needs, and it is the realistic piece: nobody
# writes "routine: Europe" in an email.
CITIES = {
    "U.S.": ["Chicago", "Dallas", "Denver", "Boston", "Seattle", "Atlanta"],
    "Canada": ["Toronto", "Vancouver", "Montreal", "Calgary"],
    "Mexico": ["Mexico City", "Guadalajara"],
    "Europe": ["London", "Paris", "Madrid", "Frankfurt", "Rome"],
    "China": ["Shanghai", "Beijing", "Guangzhou"],
    "Japan": ["Tokyo", "Osaka"],
    "India": ["Delhi", "Mumbai", "Bengaluru"],
    "South Korea": ["Seoul", "Busan"],
    "Australia": ["Sydney", "Melbourne"],
    "New Zealand": ["Auckland", "Wellington"],
    "Colombia": ["Bogota", "Medellin"],
    "Peru": ["Lima"],
    "Ecuador": ["Quito", "Guayaquil"],
    "Panama": ["Panama City"],
    "Cuba": ["Havana"],
    "Israel": ["Tel Aviv"],
    "Qatar": ["Doha"],
    "Puerto Rico": ["San Juan"],
    "South America": ["Sao Paulo", "Buenos Aires"],
}
US_HUBS = ["Chicago", "Dallas", "Charlotte", "Phoenix", "Miami", "Philadelphia"]

CABIN_PHRASE = {
    "Basic Economy": ["Basic Economy", "the basic economy fare", "a Basic Economy ticket"],
    "Main Cabin": ["Main Cabin", "a Main Cabin seat", "regular Main Cabin"],
    "Main Plus": ["Main Plus", "a Main Plus fare"],
    "Premium Economy": ["Premium Economy", "a Premium Economy seat"],
    "Business": ["Business", "a Business class seat", "business class"],
    "First": ["First", "a First class seat", "first class"],
}


def _dims(size, rng):
    a, b, c = size
    return rng.choice([
        f"{a} x {b} x {c} inches",
        f'{a}" x {b}" x {c}"',
        f"{a} by {b} by {c} inches",
        f"{a}in x {b}in x {c}in",
        f"{a} × {b} × {c} in",
    ])


def _wt(w, rng):
    return rng.choice([f"{w} lb", f"{w} lbs", f"{w} pounds",
                       f"weighs {w} pounds", f"scaled at {w} lb"])


def _bag_sentence(bag, rng, checked):
    name, d, w = bag["name"], _dims(bag["size"], rng), _wt(bag["weight"], rng)
    if not checked:
        return rng.choice([
            f"I'm keeping one {name} with me in the cabin, {d}, {w}.",
            f"One {name} ({d}, {w}) stays with me as my carry-on.",
            f"My carry-on is a {name}, {d} and {w}.",
        ])
    return rng.choice([
        f"A {name}, {d}, {w}.",
        f"One {name} measuring {d} and {w}.",
        f"{name.capitalize()}: {d}, {w}.",
        f"There's a {name} that is {d} and {w}.",
    ])


def narrate(info, seed=0):
    """One case as prose. Deterministic given (info, seed)."""
    rng = random.Random(f"{seed}|{info}")
    routine, direction = info["routine"], info["direction"]
    far = rng.choice(CITIES.get(routine, [routine]))
    if routine == "U.S.":
        hub, other = rng.sample(US_HUBS, 2)
        origin, dest = (hub, other) if direction == 0 else (other, hub)
    else:
        hub = rng.choice(US_HUBS)
        origin, dest = (hub, far) if direction == 0 else (far, hub)

    fare = info["base_price"]
    cabin = rng.choice(CABIN_PHRASE[info["customer_class"]])
    bags = info["bag_list"]
    carry, checked = bags[0], bags[1:]

    route = rng.choice([
        f"I'm flying from {origin} to {dest}.",
        f"My trip is {origin} to {dest}.",
        f"I booked {origin} → {dest}.",
        f"Itinerary: departing {origin}, arriving {dest}.",
    ])
    fare_s = rng.choice([
        f"The ticket came to ${fare} in {cabin}.",
        f"I paid ${fare} for {cabin}.",
        f"Fare was ${fare}.00, booked in {cabin}.",
        f"{cabin.capitalize()}, ${fare} for the seat.",
    ])
    lead = rng.choice([
        f"I'm checking {len(checked)} bags.",
        f"I have {len(checked)} bags to check.",
        f"Bags I'm putting in the hold: {len(checked)}.",
        f"Checking {len(checked)} pieces of luggage.",
    ])
    noise = rng.sample([
        f"Confirmation code {''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(6))}.",
        f"Seat {rng.randint(3, 38)}{rng.choice('ABCDEF')}.",
        f"Departure is {rng.randint(1, 12)}:{rng.choice(['05', '15', '30', '45'])} "
        f"{rng.choice(['am', 'pm'])}.",
        "I'd like a window if one is still open.",
        "No meal preference to note.",
        f"Booked {rng.randint(2, 9)} weeks ago.",
    ], k=rng.randint(1, 2))

    parts = [route, fare_s, _bag_sentence(carry, rng, False), lead]
    parts += [_bag_sentence(b, rng, True) for b in checked]
    parts += noise
    # keep route/fare early (an email opens with the trip) but shuffle the tail, so
    # the bag block and the noise are not always in the same place
    head, tail = parts[:2], parts[2:]
    rng.shuffle(head)
    return " ".join(head + tail)


if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, "src")
    from symbol_scramble.rulearena.loader import load_airline
    for p in load_airline(0, limit=95)[5:9]:
        print(narrate(p.entities, seed=1))
        print("   truth:", json.dumps(p.entities))
        print()
