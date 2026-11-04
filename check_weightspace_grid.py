#!/usr/bin/env python3
"""Verify completeness of all backbone x language x entity checkpoints before freezing."""
from pathlib import Path
import os
import sys
from weightspace.checkpoints import discover, VALID_LANGUAGES, VALID_ENTITIES

originals = Path(os.environ.get('ORIGINALS', str(Path.home() / 'ner_lab/TFG_Aziz/models')))
trained = Path(os.environ.get('TRAINED', '/mnt/fscratch/users/gaias_001_uma/fgd/ner_lab/TFG_Aziz/weights'))
bases, records = discover(originals, trained)
expected = {(b, l, e) for b in bases for l in VALID_LANGUAGES for e in VALID_ENTITIES}
actual = {(r.backbone, r.language, r.entity) for r in records}
missing = sorted(expected - actual)
print(f'Found {len(records)} / {len(expected)} expected checkpoints ({len(bases)} backbones, {len(VALID_LANGUAGES)} languages, {len(VALID_ENTITIES)} entities).')
if missing:
    print('Missing checkpoints:')
    for backbone, language, entity in missing:
        print(f'  {language}/{entity}/{backbone}')
    print('Do not freeze the run if these NER trainings are still pending.')
    sys.exit(2)
print('All expected checkpoints are present.')
