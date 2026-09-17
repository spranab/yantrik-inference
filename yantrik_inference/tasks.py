"""A benchmark task with an oracle.

Synthetic claim records: the facts are drawn balanced, rendered as a paragraph,
and asked 28 typed questions whose answers follow from the facts by construction.
So accuracy is exact, and the trivial baseline can be computed from the generator
rather than guessed.

That baseline matters. Always answering each field's most common value scores
about 0.60 here. A benchmark without that number cannot tell a working reader
from a broken one: an early version of this code used a plain completion prompt,
answered "yes" to every boolean, and scored 0.35 — below the baseline — which is
how the bug was found.
"""
from __future__ import annotations

import random
from typing import List, Tuple

REGIONS = ("domestic", "offshore", "regional")
CHANNELS = ("online", "in-store", "phone")
TIERS = ("bronze", "silver", "gold")
STATUSES = ("pending", "settled", "reversed")
Case = Tuple[str, List[Tuple[str, Tuple[str, ...], str]]]


def make_case(rng: random.Random) -> Case:
    f = dict(
        amount=rng.choice([rng.randint(200, 9800), rng.randint(10200, 90000)]),
        region=rng.choice(REGIONS), channel=rng.choice(CHANNELS),
        tier=rng.choice(TIERS), status=rng.choice(STATUSES),
        card_present=rng.random() < 0.5, device_new=rng.random() < 0.5,
        ip_match=rng.random() < 0.5, verified=rng.random() < 0.5,
        disputed=rng.random() < 0.5,
        chargebacks=rng.choice([0, rng.randint(1, 6)]),
        tenure=rng.choice([rng.randint(1, 11), rng.randint(13, 90)]),
        items=rng.choice([rng.randint(1, 4), rng.randint(6, 30)]),
        night=rng.random() < 0.5, weekend=rng.random() < 0.5,
    )
    yn = lambda b: "yes" if b else "no"                          # noqa: E731
    text = (
        f"Claim record. The transaction amount is {f['amount']} US dollars. "
        f"The merchant is {f['region']}. The channel was {f['channel']}. "
        f"The customer is on the {f['tier']} plan and has been a customer for "
        f"{f['tenure']} months. The payment status is {f['status']}. "
        f"The card was {'present' if f['card_present'] else 'not present'}. "
        f"The device was {'new to us' if f['device_new'] else 'one we have seen before'}. "
        f"The IP country {'matched' if f['ip_match'] else 'did not match'} the billing "
        f"country. The account is {'verified' if f['verified'] else 'unverified'}. "
        f"The customer {'has filed' if f['disputed'] else 'has not filed'} a dispute. "
        f"There have been {f['chargebacks']} prior chargebacks. "
        f"The basket held {f['items']} items. "
        f"The purchase happened {'at night' if f['night'] else 'during the day'}, "
        f"on a {'weekend' if f['weekend'] else 'weekday'}."
    )
    B = ("yes", "no")
    qs = [
        ("Is the transaction amount more than 10000 dollars?", B, yn(f["amount"] > 10000)),
        ("Is the transaction amount less than 1000 dollars?", B, yn(f["amount"] < 1000)),
        ("Was the card present?", B, yn(f["card_present"])),
        ("Was the device new to us?", B, yn(f["device_new"])),
        ("Did the IP country match the billing country?", B, yn(f["ip_match"])),
        ("Is the account verified?", B, yn(f["verified"])),
        ("Has the customer filed a dispute?", B, yn(f["disputed"])),
        ("Were there any prior chargebacks?", B, yn(f["chargebacks"] > 0)),
        ("Were there more than two prior chargebacks?", B, yn(f["chargebacks"] > 2)),
        ("Has the customer been with us for more than twelve months?", B, yn(f["tenure"] > 12)),
        ("Did the basket hold more than five items?", B, yn(f["items"] > 5)),
        ("Did the purchase happen at night?", B, yn(f["night"])),
        ("Did the purchase happen on a weekend?", B, yn(f["weekend"])),
        ("Is the merchant offshore?", B, yn(f["region"] == "offshore")),
        ("Is the merchant domestic?", B, yn(f["region"] == "domestic")),
        ("Was the channel online?", B, yn(f["channel"] == "online")),
        ("Was the channel in-store?", B, yn(f["channel"] == "in-store")),
        ("Is the customer on the gold plan?", B, yn(f["tier"] == "gold")),
        ("Is the customer on the bronze plan?", B, yn(f["tier"] == "bronze")),
        ("Has the payment settled?", B, yn(f["status"] == "settled")),
        ("Was the payment reversed?", B, yn(f["status"] == "reversed")),
        ("Is the payment still pending?", B, yn(f["status"] == "pending")),
        ("Which region is the merchant in?", REGIONS, f["region"]),
        ("Which channel was used?", CHANNELS, f["channel"]),
        ("Which plan is the customer on?", TIERS, f["tier"]),
        ("What is the payment status?", STATUSES, f["status"]),
        ("Was the card absent?", B, yn(not f["card_present"])),
        ("Is the account unverified?", B, yn(not f["verified"])),
    ]
    return text, qs


def majority_baseline(cases: List[Case]) -> float:
    """Always answer each field's most common value, computed over these cases."""
    if not cases:
        return 0.0
    cols = list(zip(*[[g for _, _, g in qs] for _, qs in cases]))
    return sum(list(c).count(max(set(c), key=list(c).count)) / len(c) for c in cols) / len(cols)
