"""Map model-space landmark JSON back through a saved ceph candidate transform.

Input JSON: [{"name": "N", "model": [x, y]}, ...].  Prints mapped points as JSON.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ceph_canonicalization import map_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transforms", type=Path)
    parser.add_argument("candidate", help="Candidate name in transforms.json")
    parser.add_argument("points", type=Path, help="JSON list of model-space landmarks")
    args = parser.parse_args()
    report = json.loads(args.transforms.read_text(encoding="utf-8-sig"))
    matches = [item for item in report["candidates"] if item["name"] == args.candidate]
    if len(matches) != 1:
        parser.error(f"Candidate {args.candidate!r} not found")
    items = json.loads(args.points.read_text(encoding="utf-8-sig"))
    original = map_points(np.asarray(matches[0]["model_to_original"], dtype=np.float64),
                          [item["model"] for item in items])
    print(json.dumps([{**item, "original": point.tolist()} for item, point in zip(items, original)], indent=2))


if __name__ == "__main__":
    main()
