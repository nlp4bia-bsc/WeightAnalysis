from pathlib import Path
import gzip
import shutil
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(
    "/mnt/fscratch/users/gaias_001_uma/fgd/"
    "ner_lab/TFG_Aziz/weight_analysis_trial460"
)
OUT = Path.home() / "revision_weightspace_460"
OUT.mkdir(exist_ok=True)

is_scalar = lambda t: (
    pa.types.is_integer(t)
    or pa.types.is_floating(t)
    or pa.types.is_boolean(t)
    or pa.types.is_string(t)
    or pa.types.is_large_string(t)
)

report = []

for path in sorted(BASE.glob("*.parquet")):
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    cols = [f.name for f in schema if is_scalar(f.type)]

    # Las comparaciones grandes se reducen al nivel global
    # del encoder. El resto conserva todas las filas escalares.
    only_encoder = path.name.startswith(
        ("03_", "04_", "05_")
    ) and "level" in cols

    dest = OUT / (path.stem + ".csv.gz")
    kept = 0
    header = True

    with gzip.open(dest, "wt", encoding="utf-8") as handle:
        for batch in pf.iter_batches(
            batch_size=30000, columns=cols
        ):
            df = batch.to_pandas()

            if only_encoder:
                df = df[df["level"] == "encoder"]

            if df.empty:
                continue

            df.to_csv(handle, index=False, header=header)
            header = False
            kept += len(df)

    if kept == 0:
        dest.unlink()

    report.append(
        f"{path.name}\n"
        f"  Filas originales: {pf.metadata.num_rows}\n"
        f"  Filas exportadas: {kept}\n"
        f"  Solo encoder: {only_encoder}\n"
        f"  Columnas: {', '.join(schema.names)}\n"
    )

(OUT / "inventario.txt").write_text(
    "\n".join(report), encoding="utf-8"
)

archive = shutil.make_archive(
    str(Path.home() / "revision_weightspace_460"),
    "zip",
    root_dir=OUT
)

print("Archivo generado:", archive)
