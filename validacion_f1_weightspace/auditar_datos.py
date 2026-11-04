import os
from pathlib import Path
import pandas as pd

base = Path(os.environ["WS_DATA"])
f1_path = Path(os.environ["WS_F1"])

files = [
    "01_pretraining_drift.parquet",
    "02_spectral_fingerprints.parquet",
    "03_cross_language.parquet",
    "04_cross_entity.parquet",
    "05_cross_language_entity.parquet",
    "07_cross_backbone.parquet",
    "08_language_entity_decomposition.parquet",
    "09_cross_backbone_by_entity.parquet",
    "11_model_centrality.parquet",
]

print("# AUDITORÍA DE DATOS\n")

for filename in files:
    path = base / filename
    print(f"\n## {filename}")

    if not path.exists():
        print("FALTA EL ARCHIVO")
        continue

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)

    print("Filas:", pf.metadata.num_rows)
    print("Grupos de filas:", pf.metadata.num_row_groups)
    print("Columnas:", pf.schema_arrow.names)

f1 = pd.read_parquet(f1_path)
f1 = f1.rename(columns={"model": "backbone"})

key = ["language", "entity", "backbone"]

print("\n# EVALUACIÓN NER")
print("Filas:", len(f1))
print("Duplicados:", f1.duplicated(key).sum())
print("Folds:")
print(f1["n_folds"].value_counts(dropna=False))

print("\nCobertura idioma/entidad:")
print(
    f1.groupby(["language", "entity"])
    .size()
    .to_string()
)

print("\nF1 ausente:")
print(f1[f1["strict_f1_mean"].isna()][key])

w = pd.read_parquet(
    base / "01_pretraining_drift.parquet",
    columns=key + ["level", "fold"]
)

w = w[w["level"] == "encoder"]
w = w.drop_duplicates(key + ["fold"])

print("\n# CHECKPOINTS DE PESOS")
print("Checkpoints encoder:", len(w))
print("Folds:", w["fold"].value_counts().to_dict())

join = w.merge(
    f1[key],
    on=key,
    how="outer",
    indicator=True
)

print("\nCoincidencias:")
print(join["_merge"].value_counts())

print("\nNo emparejados:")
print(join[join["_merge"] != "both"].to_string(index=False))

pairs = pd.read_parquet(
    base / "07_cross_backbone.parquet",
    columns=[
        "language", "entity",
        "backbone_a", "backbone_b"
    ]
)

print("\n# COMPARACIONES ENTRE BACKBONES")
print("Pares:", len(pairs))

dup = pairs.duplicated([
    "language", "entity",
    "backbone_a", "backbone_b"
])

print("Pares orientados duplicados:", dup.sum())

print("\nAuditoría finalizada.")
