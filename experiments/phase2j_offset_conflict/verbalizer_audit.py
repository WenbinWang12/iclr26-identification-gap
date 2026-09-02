"""Audit Order-4 verbalizer sharing at the token level, not the string level.

The theory note's sharing table was built from label *strings*.  What actually
couples the task-agnostic offset b is sharing of the **first decoded piece**,
because that is the coordinate b acts on.  These can differ: two tasks whose
label strings look distinct may still collide on a first piece, and a label that
looks shared may tokenize differently in different label-set contexts (it does
not here, T5 tokenizes labels independently, but that must be checked, not
assumed).

Also flags any task whose labels do NOT have distinct first pieces -- such a
task is outside the formalism and the probe must refuse to score it.

Writes JSON; reads no data files, only the pinned label sets and the tokenizer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HOME", "/mnt/data/wenbin/iclr26/models/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from experiments.phase2i_anchored_cvar import order4_data as od  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google-t5/t5-large")
    parser.add_argument("--out", default="/mnt/data/wenbin/iclr26/runs/phase2j_verbalizer_audit.json")
    # Yelp and Amazon are known-degenerate and documented as excluded by
    # precondition (protocol Amendment A1).  Only an UNEXPECTED degenerate task
    # is a failure, so the known pair is passed in rather than hard-coded as OK.
    parser.add_argument("--expected-degenerate", default="Yelp,Amazon")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False)

    report: dict[str, object] = {"model": args.model}
    per_task = {}
    first_piece_owners: dict[int, list[str]] = {}
    string_owners: dict[str, list[str]] = {}
    degenerate = []

    for index, task in enumerate(od.ORDER4_TASKS, start=1):
        entry = {"order_position": index, "category": task.category,
                 "dataset": task.dataset, "labels": {}}
        firsts = []
        for label in task.labels:
            ids = [int(i) for i in tokenizer(label, add_special_tokens=False)["input_ids"]]
            entry["labels"][label] = {"ids": ids, "n_pieces": len(ids),
                                      "first_piece": ids[0],
                                      "first_piece_text": tokenizer.convert_ids_to_tokens([ids[0]])[0]}
            firsts.append(ids[0])
            first_piece_owners.setdefault(ids[0], []).append("%s:%s" % (task.name, label))
            string_owners.setdefault(label, []).append(task.name)
        entry["first_pieces_distinct"] = len(set(firsts)) == len(firsts)
        if not entry["first_pieces_distinct"]:
            degenerate.append(task.name)
        per_task[task.name] = entry

    report["per_task"] = per_task
    report["degenerate_tasks"] = degenerate
    report["shared_first_pieces"] = {
        str(piece): sorted(owners)
        for piece, owners in sorted(first_piece_owners.items())
        if len({owner.split(":")[0] for owner in owners}) > 1
    }
    report["shared_label_strings"] = {
        label: sorted(set(tasks))
        for label, tasks in sorted(string_owners.items())
        if len(set(tasks)) > 1
    }
    # A collision that string-level analysis would miss: same first piece,
    # different label strings.  These couple b without looking shared.
    hidden = {}
    for piece, owners in first_piece_owners.items():
        labels = {owner.split(":", 1)[1] for owner in owners}
        tasks = {owner.split(":")[0] for owner in owners}
        if len(labels) > 1 and len(tasks) > 1:
            hidden[str(piece)] = sorted(owners)
    report["hidden_collisions_same_first_piece_different_strings"] = hidden

    expected = {n.strip() for n in args.expected_degenerate.split(",") if n.strip()}
    unexpected = sorted(set(degenerate) - expected)
    absent = sorted(expected - set(degenerate))
    report["expected_degenerate"] = sorted(expected)
    report["unexpected_degenerate"] = unexpected
    # If a task we expected to be degenerate no longer is, the tokenizer or the
    # pinned label set changed underneath us; that also invalidates Amendment A1.
    report["expected_degenerate_not_reproduced"] = absent

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "degenerate_tasks": degenerate,
        "unexpected_degenerate": unexpected,
        "expected_degenerate_not_reproduced": absent,
        "n_shared_first_pieces": len(report["shared_first_pieces"]),
        "n_shared_strings": len(report["shared_label_strings"]),
        "hidden_collisions": hidden,
    }, indent=2), flush=True)
    ok = not unexpected and not absent
    print("VERBALIZER_AUDIT %s out=%s" % ("PASS" if ok else "FAIL", args.out), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
