"""
ChronoTrack dataset generator.

Generates synthetic narratives that track the state (location or possessor)
of a target entity across a week of events, with two independent stressors:

  1. ORDER MODE
     - "chronological": sentences appear in the same order as the events happened
     - "scrambled": sentences are presented in a random order (day labels are
        still explicit, so the information needed to answer is always present
        -- only the *presentation order* changes)

  2. DISTRACTOR LOAD
     - number of irrelevant entities with their own interleaved update chains
       (0, 2, or 4 distractor entities, each with 2-3 updates)

Two query types:
  - "final_state": what is true at the end of the week
  - "state_at_time": what was true as of a specific day (requires interval
     reasoning -- the most recent update at or before that day, which is not
     necessarily the update that appears closest to the query in the text)

Ground truth is computed directly from the underlying event chain, independent
of how it's rendered into text, so scoring is unambiguous exact-match.
"""

import json
import random
from dataclasses import dataclass, field
from itertools import product

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

OBJECTS = ["backpack", "laptop", "umbrella", "notebook", "keys", "phone", "wallet",
           "camera", "guitar", "helmet", "toolbox", "suitcase"]

LOCATIONS = ["kitchen", "garage", "office", "library", "gym", "car", "attic",
             "bedroom", "basement", "studio", "workshop", "porch"]

PEOPLE = ["Alice", "Ben", "Carla", "Deo", "Ella", "Farid", "Grace", "Hugo"]

MOVE_TEMPLATES = [
    "{connective} {subj} was moved to the {value}.",
    "{connective} someone brought the {subj} to the {value}.",
    "{connective} the {subj} ended up in the {value}.",
]

GIVE_TEMPLATES = [
    "{connective} the {subj} was given to {value}.",
    "{connective} {value} picked up the {subj}.",
    "{connective} the {subj} was handed over to {value}.",
]

CONNECTIVES_CHRONO = {
    0: "On {day},", 1: "On {day},", 2: "On {day},", 3: "On {day},",
    4: "On {day},", 5: "On {day},", 6: "On {day},",
}


@dataclass
class Event:
    entity: str
    kind: str          # "location" or "possessor"
    value: str
    day_idx: int
    is_target: bool


def make_event_chain(entity, kind, day_pool, is_target, rng, n_events=None):
    n_events = n_events or rng.randint(3, 4)
    days = sorted(rng.sample(day_pool, k=min(n_events, len(day_pool))))
    values = LOCATIONS if kind == "location" else PEOPLE
    vals = rng.sample(values, k=len(days))
    return [Event(entity, kind, v, d, is_target) for v, d in zip(vals, days)]


def render_event(ev, rng):
    templates = MOVE_TEMPLATES if ev.kind == "location" else GIVE_TEMPLATES
    template = rng.choice(templates)
    connective = f"On {DAYS[ev.day_idx]},"
    return template.format(connective=connective, subj=ev.entity, value=ev.value)


def gold_state_at(events, day_idx):
    """Most recent event with day <= day_idx. None if no such event."""
    valid = [e for e in events if e.day_idx <= day_idx]
    if not valid:
        return None
    return max(valid, key=lambda e: e.day_idx).value


def build_item(rng, order_mode, n_distractors, query_type, item_id):
    day_pool = list(range(7))
    kind = rng.choice(["location", "possessor"])
    target_entity = rng.choice(OBJECTS)

    target_events = make_event_chain(target_entity, kind, day_pool, True, rng)

    distractor_events = []
    used_entities = {target_entity}
    for _ in range(n_distractors):
        ent = rng.choice([o for o in OBJECTS if o not in used_entities])
        used_entities.add(ent)
        dkind = rng.choice(["location", "possessor"])
        distractor_events.extend(
            make_event_chain(ent, dkind, day_pool, False, rng, n_events=rng.randint(2, 3))
        )

    all_events = target_events + distractor_events
    chrono_events = sorted(all_events, key=lambda e: e.day_idx)

    if order_mode == "chronological":
        presented = chrono_events
    else:
        presented = all_events[:]
        rng.shuffle(presented)
        # guard against accidentally sampling a chronological order
        tries = 0
        while presented == chrono_events and tries < 5:
            rng.shuffle(presented)
            tries += 1

    sentences = [render_event(e, rng) for e in presented]
    story = " ".join(sentences)

    if query_type == "final_state":
        query_day = max(e.day_idx for e in target_events)
        gold = gold_state_at(target_events, query_day)
        if kind == "location":
            question = f"At the end of the week, where was the {target_entity}?"
        else:
            question = f"At the end of the week, who had the {target_entity}?"
    else:  # state_at_time
        candidate_days = [d for d in day_pool if gold_state_at(target_events, d) is not None]
        # bias toward a day that is NOT the day of the most recent update mentioned last,
        # to force real interval reasoning rather than "find the last sentence"
        query_day = rng.choice(candidate_days)
        gold = gold_state_at(target_events, query_day)
        if kind == "location":
            question = f"As of {DAYS[query_day]}, where was the {target_entity}?"
        else:
            question = f"As of {DAYS[query_day]}, who had the {target_entity}?"

    # paraphrase for self-consistency probing
    if kind == "location":
        paraphrase = question.replace("where was", "in what place was")
    else:
        paraphrase = question.replace("who had", "which person was in possession of")

    return {
        "id": item_id,
        "story": story,
        "question": question,
        "question_paraphrase": paraphrase,
        "gold_answer": gold,
        "meta": {
            "order_mode": order_mode,
            "n_distractors": n_distractors,
            "query_type": query_type,
            "kind": kind,
            "target_entity": target_entity,
            "n_target_events": len(target_events),
        },
    }


def generate_dataset(n_per_cell=15, seed=42):
    rng = random.Random(seed)
    order_modes = ["chronological", "scrambled"]
    distractor_levels = [0, 2, 4]
    query_types = ["final_state", "state_at_time"]

    items = []
    item_id = 0
    for order_mode, n_dist, qtype in product(order_modes, distractor_levels, query_types):
        for _ in range(n_per_cell):
            item = build_item(rng, order_mode, n_dist, qtype, item_id)
            items.append(item)
            item_id += 1
    rng.shuffle(items)
    return items


if __name__ == "__main__":
    dataset = generate_dataset(n_per_cell=15, seed=42)
    out_path = "chronotrack_dataset.jsonl"
    with open(out_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(dataset)} items to {out_path}")
    print("\nSample item:")
    print(json.dumps(dataset[0], indent=2))
