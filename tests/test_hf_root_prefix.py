"""Regressions for actual Picasso BAAI bare AutoModel checkpoint naming."""
import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from weightspace.checkpoints import WeightReader, audit_alignment, normalize
from weightspace.cli import run


def make_dir(path: Path, model_type: str, tensors: dict):
    path.mkdir(parents=True)
    save_file(tensors, path / 'model.safetensors')
    (path / 'config.json').write_text(json.dumps({
        'model_type': model_type, 'hidden_size': 4,
        'num_hidden_layers': 1, 'num_attention_heads': 2,
        'vocab_size': 7,
    }))


def test_baai_automodel_to_tokenclassification_prefix_and_buffers(tmp_path):
    """Bare encoder and bert.encoder align; position_ids is an integer buffer."""
    original = tmp_path / 'models' / 'BAAI--bge-large-en-v1.5'
    trained = tmp_path / 'weights' / 'cz' / 'disease' / original.name
    tensors = {
        'embeddings.word_embeddings.weight': torch.randn(7, 4),
        'embeddings.position_embeddings.weight': torch.randn(5, 4),
        'embeddings.position_ids': torch.arange(5, dtype=torch.long).reshape(1, 5),
        'encoder.layer.0.attention.self.query.weight': torch.randn(4, 4),
        'encoder.layer.0.attention.self.query.bias': torch.randn(4),
        'pooler.dense.weight': torch.randn(4, 4),
    }
    # The final model drops the pooler, omits position_ids, adds classifier.
    updated = {f'bert.{k}': v + .05 for k, v in tensors.items()
               if k != 'embeddings.position_ids' and not k.startswith('pooler.')}
    updated['classifier.weight'] = torch.randn(3, 4)
    updated['classifier.bias'] = torch.randn(3)
    make_dir(original, 'bert', tensors)
    make_dir(trained, 'bert', updated)
    with WeightReader(trained) as f, WeightReader(original) as b:
        keys, result = audit_alignment(f, b)
        assert result['element_coverage'] == 1
        assert len(keys) == 4
        assert 'bert.encoder.layer.0.attention.self.query.weight' in keys
        assert 'bert.embeddings.word_embeddings.weight' in keys
        assert 'bert.embeddings.position_ids' in b.keys
        assert f.raw_names['bert.embeddings.word_embeddings.weight'] == 'bert.embeddings.word_embeddings.weight'
        assert b.raw_names['bert.embeddings.word_embeddings.weight'] == 'embeddings.word_embeddings.weight'
        assert 'bert.pooler.dense.weight' not in keys
    # Deep dry-run is feasible without GPU; it should not write Parquet.
    run(['--originals', str(original.parent), '--trained', str(tmp_path / 'weights'),
         '--output', str(tmp_path / 'results'), '--deep-dry-run'])
    assert not (tmp_path / 'results').exists()


@pytest.mark.parametrize('model_type,expected_prefix', [
    ('bert','bert'),('roberta','roberta'),('xlm-roberta','roberta'),
    ('camembert','roberta'),('distilbert','distilbert'),('electra','electra'),
    ('deberta','deberta'),('deberta-v2','deberta'),('albert','albert'),
])
def test_only_explicit_family_namespace_is_mapped(model_type, expected_prefix):
    assert normalize('encoder.layer.0.attention.self.query.weight', model_type) == (
        f'{expected_prefix}.encoder.layer.0.attention.self.query.weight')
    assert normalize('embeddings.LayerNorm.bias', model_type) == f'{expected_prefix}.embeddings.LayerNorm.bias'
    assert normalize(f'{expected_prefix}.encoder.layer.0.weight',model_type) == (
        f'{expected_prefix}.encoder.layer.0.weight')
    assert normalize('encoder.layer.0.weight', 'unknown-architecture') == 'encoder.layer.0.weight'


def test_no_implicit_cross_family_match(tmp_path):
    base = tmp_path/'base'; trained = tmp_path/'trained'
    make_dir(base, 'roberta', {'encoder.layer.0.attention.self.query.weight': torch.eye(4)})
    make_dir(trained, 'bert', {'bert.encoder.layer.0.attention.self.query.weight': torch.eye(4)})
    with WeightReader(trained) as f, WeightReader(base) as b:
        with pytest.raises(ValueError, match='Invalid base/train alignment'):
            audit_alignment(f, b)


def test_ambiguous_root_alias_fails_not_silently_overwrites(tmp_path):
    base = tmp_path/'base'
    make_dir(base, 'bert', {'encoder.layer.0.weight': torch.ones((2,2)),
                            'bert.encoder.layer.0.weight': torch.zeros((2,2))})
    with pytest.raises(ValueError, match='Normalized-key collision'):
        with WeightReader(base):
            pass


def test_matching_names_different_shapes_are_rejected(tmp_path):
    base=tmp_path/'base'; fin=tmp_path/'fin'
    make_dir(base,'bert',{'encoder.layer.0.attention.self.query.weight':torch.ones((4,4))})
    make_dir(fin,'bert',{'bert.encoder.layer.0.attention.self.query.weight':torch.ones((3,4))})
    with WeightReader(fin) as f, WeightReader(base) as b:
        with pytest.raises(ValueError,match='mismatch'):
            audit_alignment(f,b)
