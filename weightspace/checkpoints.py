"""Read local HF weights, and reject ambiguous or ungrounded tensor mappings.

Never calls Hugging Face Hub, instantiates a model or performs inference.
"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import re
from typing import Iterator
import torch
from safetensors import safe_open

LOG = logging.getLogger(__name__)
EXTENSIONS = ('model.safetensors.index.json', 'model.safetensors', 'pytorch_model.bin.index.json', 'pytorch_model.bin')
BACKBONES = ('bert', 'roberta', 'xlm_roberta', 'deberta', 'deberta_v2', 'distilbert', 'electra', 'camembert', 'albert')
COMPONENTS = ('q', 'k', 'v', 'o', 'ffn_up', 'ffn_down', 'ffn_gate', 'norm', 'embeddings', 'other')
SPECTRAL_COMPONENTS = ('q', 'k', 'v', 'o', 'ffn_up', 'ffn_down', 'ffn_gate')
# Restrict scanning to the actual MultiClinAI experimental design.
# In particular, weights/.overwrite_v2 contains backups/administrative files,
# NOT checkpoints: do not traverse or delete them.
VALID_LANGUAGES = frozenset({'cz', 'en', 'es', 'it', 'nl', 'ro', 'sv'})
VALID_ENTITIES = frozenset({'disease', 'procedure', 'symptom'})

@dataclass(frozen=True, order=True)
class Checkpoint:
    backbone: str
    language: str
    entity: str
    path: Path
    fold: int = 1
    @property
    def uid(self) -> str:
        return f'{self.backbone}::{self.language}::{self.entity}::{self.fold}'


def weight_files(directory: Path) -> list[Path]:
    """Return recognized checkpoint(s), preferring safe formats; error if ambiguous."""
    safe_idx = directory / EXTENSIONS[0]
    safe_one = directory / EXTENSIONS[1]
    pt_idx = directory / EXTENSIONS[2]
    pt_one = directory / EXTENSIONS[3]
    if safe_idx.is_file(): return [safe_idx]
    if safe_one.is_file(): return [safe_one]
    if pt_idx.is_file(): return [pt_idx]
    if pt_one.is_file(): return [pt_one]
    return []


def discover(originals: Path, trained: Path) -> tuple[dict[str, Path], list[Checkpoint]]:
    """Discover only weights/<language>/<entity>/<backbone> checkpoint directories.

    Restrict the top-level languages and entities to the known MultiClinAI
    labels; never recurse into hidden backups such as `.overwrite_v2`.
    Within a valid language/entity, a visible model directory without weights
    remains a hard error so failed or partial training is not ignored.
    """
    if not originals.is_dir() or not trained.is_dir():
        raise FileNotFoundError(f'Originals={originals} and trained={trained} must be directories')

    bases = {
        d.name: d for d in sorted(originals.iterdir())
        if d.is_dir() and not d.name.startswith('.') and weight_files(d)
    }
    records: list[Checkpoint] = []
    invalid: list[str] = []
    for language in sorted(trained.iterdir()):
        if not language.is_dir() or language.name.startswith('.'):
            continue
        lang = language.name.lower()
        if lang not in VALID_LANGUAGES:
            LOG.warning('Skipping non-MultiClinAI directory under weights/: %s', language)
            continue
        for entity in sorted(language.iterdir()):
            if not entity.is_dir() or entity.name.startswith('.'):
                continue
            label = entity.name.lower()
            if label not in VALID_ENTITIES:
                raise ValueError(f'Unexpected entity directory for {lang}: {entity}')
            for model in sorted(entity.iterdir()):
                if not model.is_dir() or model.name.startswith('.'):
                    continue
                if weight_files(model):
                    records.append(Checkpoint(model.name, lang, label, model))
                else:
                    invalid.append(str(model))
    if not records:
        raise ValueError(f'No trained checkpoints in {trained}/<language>/<entity>/<backbone>')
    if invalid:
        raise ValueError(f'Model dirs with no recognized weights: {invalid[:20]}')
    unknown = sorted({x.backbone for x in records if x.backbone not in bases})
    if unknown:
        raise ValueError(f'Trained backbones without ORIGINAL weights: {unknown}')
    if len({x.uid for x in records}) != len(records):
        raise ValueError('Duplicate checkpoint IDs')
    return bases, sorted(records)


class WeightReader:
    """Safe, lazy CPU access to tensors with sharded HF index support."""
    def __init__(self, path: Path):
        self.path = path
        # HF AutoModel base checkpoints often store bare 'encoder.*' keys,
        # while AutoModelForTokenClassification stores 'bert.encoder.*'.
        # The actual HF model_type supplies the only permitted namespace alias.
        self.model_type = config_metadata(path).get('model_type')
        self.stack = ExitStack()
        self.handles: dict[Path, object] = {}
        self.pt: dict[Path, dict] = {}
        self.mapping: dict[str, tuple[Path,str]] = {}
        self.raw_names: dict[str,str] = {}

    def __enter__(self):
        selected = weight_files(self.path)
        if not selected: raise FileNotFoundError(f'No weights in {self.path}')
        selection = selected[0]
        if selection.suffix == '.json':
            data = json.loads(selection.read_text(encoding='utf8'))
            mapping = data.get('weight_map')
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f'Bad HF index: {selection}')
            for key, name in mapping.items():
                if Path(name).name != name: raise ValueError(f'Unsafe shard name: {name}')
                file = self.path / name
                if not file.is_file(): raise FileNotFoundError(file)
                self._add(key, file)
            # Verify index names exist in referenced shards before consuming any tensor.
            for f in sorted(set(p for p,_ in self.mapping.values())):
                actual = set(self._keys(f))
                required = {raw for file,raw in self.mapping.values() if file == f}
                if not required <= actual: raise ValueError(f'Shard index mismatch: {f}: {required-actual}')
        else:
            for key in self._keys(selection): self._add(key, selection)
        if not self.mapping: raise ValueError(f'No tensors found in {self.path}')
        return self

    def _add(self, raw: str, file: Path):
        normalized = normalize(raw, self.model_type)
        if normalized in self.mapping: raise ValueError(f'Normalized-key collision: {normalized}, {self.path}')
        self.mapping[normalized] = (file, raw)
        self.raw_names[normalized] = raw

    def _keys(self, file: Path):
        if file.suffix == '.safetensors':
            if file not in self.handles:
                self.handles[file] = self.stack.enter_context(safe_open(str(file), framework='pt', device='cpu'))
            return self.handles[file].keys()
        if file not in self.pt:
            try: obj = torch.load(file, map_location='cpu', weights_only=True, mmap=True)
            except (TypeError, RuntimeError, ValueError):
                obj = torch.load(file, map_location='cpu', weights_only=True)
            for name in ('state_dict','model_state_dict','model'):
                if isinstance(obj,dict) and isinstance(obj.get(name),dict): obj = obj[name]; break
            if not isinstance(obj, dict): raise ValueError(f'Expected a state_dict: {file}')
            self.pt[file] = {str(k):v for k,v in obj.items() if isinstance(v,torch.Tensor)}
        return self.pt[file].keys()

    def shape_dtype(self, key: str) -> tuple[tuple[int, ...], bool]:
        """Get tensor shape and floating-point flag without materializing safetensors."""
        file, raw = self.mapping[key]
        self._keys(file)
        if file.suffix == '.safetensors':
            sl = self.handles[file].get_slice(raw)
            return tuple(sl.get_shape()), sl.get_dtype() in ('F16', 'F32', 'F64', 'BF16')
        tensor = self.pt[file][raw]
        return tuple(tensor.shape), tensor.is_floating_point()

    @property
    def keys(self): return set(self.mapping)

    def get(self, key: str) -> torch.Tensor:
        file, raw = self.mapping[key]
        self._keys(file)
        t = self.handles[file].get_tensor(raw) if file.suffix == '.safetensors' else self.pt[file][raw]
        return t.detach().cpu()

    def __exit__(self, *_):
        self.stack.close(); self.handles.clear(); self.pt.clear(); self.mapping.clear(); self.raw_names.clear()


# Config-aware namespace used by Hugging Face AutoModelForTokenClassification.
# This is NOT an alias between families: 'bert' is never mapped to 'roberta'.
# HF AutoModel often omits its own root module prefix from state_dict keys.
HF_ENCODER_PREFIX = {
    'bert': 'bert', 'roberta': 'roberta', 'xlm-roberta': 'roberta',
    'camembert': 'roberta', 'electra': 'electra', 'distilbert': 'distilbert',
    'deberta': 'deberta', 'deberta-v2': 'deberta', 'albert': 'albert',
}
# Only these top-level components are known to occur both bare on AutoModel
# checkpoints and prefixed on AutoModelForTokenClassification checkpoints.
BARE_ENCODER_COMPONENTS = frozenset({'embeddings', 'encoder', 'pooler', 'transformer'})


def normalize(raw: str, model_type: str | None = None) -> str:
    """Canonical HF encoder key, preserving architecture family identity.

    Known transport wrappers are stripped first. When config.json identifies a
    supported family, an HF base-model key such as 'encoder.layer.0...' is
    mapped to 'bert.encoder.layer.0...' for model_type='bert'. A key that
    already begins with 'bert.' is unchanged. Unsupported architectures are
    deliberately not guessed. Key collisions are rejected by WeightReader.
    """
    key = raw
    while True:
        prior = key
        for prefix in ('module.', 'base_model.model.', 'base_model.', 'model.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        if key == prior:
            break
    root = HF_ENCODER_PREFIX.get(model_type or '')
    if root and key.split('.', 1)[0] in BARE_ENCODER_COMPONENTS:
        return f'{root}.{key}'
    return key


def identify(key: str, shape: tuple[int,...]) -> tuple[int | None,str,bool]:
    """Identify conservative common encoder components; unknowns kept for auditing."""
    low = key.lower()
    if re.search(r'(^|\.)(classifier|score|qa_outputs|crf|lm_head)(\.|$)',low):
        return None,'ner_head',True
    match = re.search(r'(?:encoder\.layer|transformer\.layer|transformer\.h|layers|block|layer)\.(\d+)\.',low)
    layer = int(match.group(1)) if match else None
    if any(t in low for t in ('embeddings.', 'embed_tokens.')): return layer, 'embeddings',False
    if ('layernorm' in low or 'layer_norm' in low or re.search(r'(^|\.)norm\.',low)):
        return layer,'norm',False
    rules = [
        (r'(?:^|\.)(?:query|q_proj|q_lin)\.', 'q'),
        (r'(?:^|\.)(?:key|k_proj|k_lin)\.', 'k'),
        (r'(?:^|\.)(?:value|v_proj|v_lin)\.', 'v'),
        (r'(?:attention\.output\.dense|self_attn\.out_proj|o_proj|attention\.out_lin)\.', 'o'),
        (r'(?:intermediate\.dense|fc1|up_proj|ffn\.lin1)\.', 'ffn_up'),
        (r'(?:output\.dense|fc2|down_proj|ffn\.lin2)\.', 'ffn_down'),
        (r'(?:gate_proj)\.', 'ffn_gate'),
    ]
    for rx,component in rules:
        if re.search(rx,low):
            # Only assign 'output.dense' to FFN when it is not attention.output.dense.
            if component == 'ffn_down' and 'attention.output.dense.' in low: continue
            return layer,component,False
    return layer,'other',False


def config_metadata(path: Path) -> dict:
    conf = path/'config.json'
    if not conf.is_file(): return {}
    try:
        data = json.loads(conf.read_text(encoding='utf8'))
        return {k:data.get(k) for k in ('model_type','hidden_size','num_hidden_layers','num_attention_heads','vocab_size','architectures')}
    except (ValueError, OSError): return {}


def audit_alignment(r: WeightReader, b: WeightReader, min_fraction: float = 0.99) -> tuple[list[str],dict]:
    """Reject missing encoder tensors by ELEMENT coverage; no head subtraction.

    Config/name matching cannot prove an identical original checkpoint; folder
    provenance must still be confirmed by the training setup.
    """
    keys = []; missing = []; mismatches=[]; matched_n=0; candidate_n=0
    for key in sorted(r.keys):
        shape, floating = r.shape_dtype(key)
        if not floating: continue  # HF buffers such as position_ids are NOT weights.
        layer,component,head = identify(key,shape)
        if head: continue
        # Keep unknown floating encoder components for explicit validation.
        candidate_n += math.prod(shape)
        if key not in b.keys:
            missing.append(key); continue
        w0_shape, w0_floating = b.shape_dtype(key)
        if w0_shape != shape or not w0_floating:
            mismatches.append(key); continue
        keys.append(key); matched_n += math.prod(shape)
    coverage = matched_n / candidate_n if candidate_n else 0
    status = dict(matched_elements=matched_n,candidate_elements=candidate_n,element_coverage=coverage,
                  missing_keys=missing,mismatched_keys=mismatches,n_matched_tensors=len(keys))
    if not keys or coverage < min_fraction:
        raise ValueError(f'Invalid base/train alignment ({coverage:.2%} elements); missing={missing[:8]}, mismatch={mismatches[:8]}')
    return keys,status


def get_delta(r: WeightReader, b: WeightReader, key: str) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    fin, ini = r.get(key), b.get(key)
    if fin.shape != ini.shape or not fin.is_floating_point() or not ini.is_floating_point():
        raise ValueError(f'Incompatible parameter {key}')
    fin = fin.to(torch.float64); ini = ini.to(torch.float64)
    if not torch.isfinite(fin).all() or not torch.isfinite(ini).all():
        raise ValueError(f'Nonfinite weights for {key}')
    return fin,ini,fin-ini

# Transformer multi-head attention layout: Vaswani et al. (2017),
# Attention Is All You Need, NeurIPS. https://arxiv.org/abs/1706.03762
# Standard *BERT-family* unfused Q/K/V projection output rows represent
# concatenated heads; output projection input columns correspond to heads.
# This is a structural tensor slice, NOT evidence of head functionality.
def head_views(w:torch.Tensor,component:str,config:dict):
    """Yield (head_index, matrix) only where HF encoder head layout is known.

    Conservative: skip fused qkv, grouped-query, adapters, biases and models
    with unknown attention-head weight conventions.
    """
    if w.ndim!=2 or component not in ('q','k','v','o'):return
    if config.get('model_type') not in ('bert','roberta','xlm-roberta','distilbert','electra','camembert'):
        return
    heads=config.get('num_attention_heads')
    if not isinstance(heads,int) or heads<1:return
    rows,cols=w.shape
    size=rows if component in ('q','k','v') else cols
    if size%heads:return
    for h in range(heads):
        start,end=h*(size//heads),(h+1)*(size//heads)
        view=w[start:end,:] if component in ('q','k','v') else w[:,start:end]
        yield h,view
