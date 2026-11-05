"""Check the historical experiment grid without creating a frozen run or loading tensors."""

from __future__ import annotations

import argparse
import os
from itertools import product
from pathlib import Path

from ..core import VALID_ENTITIES, VALID_LANGUAGES, discover


def check_grid(originals, trained):
    bases, records = discover(originals, trained)
    expected = set(product(bases, VALID_LANGUAGES, VALID_ENTITIES))
    actual = {(r.backbone, r.language, r.entity) for r in records}
    missing = sorted(expected - actual)
    return dict(found=len(actual), expected=len(expected), backbones=len(bases), missing=missing)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--originals",
        default=os.environ.get("ORIGINALS", str(Path.home() / "ner_lab/TFG_Aziz/models")),
    )
    parser.add_argument(
        "--trained", default=os.environ.get("TRAINED"), required="TRAINED" not in os.environ
    )
    args = parser.parse_args(argv)
    result = check_grid(args.originals, args.trained)
    print(
        f"Found {result['found']} / {result['expected']} expected checkpoints ({result['backbones']} backbones)."
    )
    for backbone, language, entity in result["missing"]:
        print(f"  MISSING {language}/{entity}/{backbone}")
    return 2 if result["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
