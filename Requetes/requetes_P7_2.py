import os
import sys
from pathlib import Path
from pymongo import MongoClient
import polars as pl


DB_NAME = "P7MLO"
COLL_NAME = "listings"

OUTDIR = os.getenv("OUTDIR", "/app/outputs")
TOP_N = int(os.getenv("TOP_N", "5"))
NETTOYER_OUTPUTS = os.getenv("NETTOYER_OUTPUTS", "0") == "1"

MONGO_URI = os.getenv("MONGO_URI", "").strip()
MONGO_HOST = os.getenv("MONGO_HOST", "localhost").strip()
MONGO_PORT = os.getenv("MONGO_PORT", "27017").strip()
MONGO_AUTH_SOURCE = os.getenv("MONGO_AUTH_SOURCE", "admin").strip()
MONGO_USER = os.getenv("MONGO_USER", "").strip()
MONGO_PASS = os.getenv("MONGO_PASS", "").strip()

PROJECTION = {
    "_id": 0,
    "last_scraped": 1,
    "room_type": 1,
    "availability_30": 1,
    "number_of_reviews": 1,
    "host_is_superhost": 1,
    "neighbourhood_cleansed": 1,
}


def die(msg: str, code: int = 1):
    print(f"[ERREUR] {msg}", file=sys.stderr)
    sys.exit(code)


def build_mongo_uri() -> str:
    if MONGO_URI:
        return MONGO_URI

    if not MONGO_USER and not MONGO_PASS:
        return f"mongodb://{MONGO_HOST}:{MONGO_PORT}"

    if not MONGO_USER or not MONGO_PASS:
        die("Il manque MONGO_USER ou MONGO_PASS.")

    return (
        f"mongodb://{MONGO_USER}:{MONGO_PASS}"
        f"@{MONGO_HOST}:{MONGO_PORT}/?authSource={MONGO_AUTH_SOURCE}"
    )


def nettoyer_outputs(outdir: str):
    p = Path(outdir)
    if not p.exists():
        return
    for f in p.glob("*.csv"):
        try:
            f.unlink()
        except Exception:
            pass


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    if NETTOYER_OUTPUTS:
        nettoyer_outputs(OUTDIR)

    uri = build_mongo_uri()

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except Exception as e:
        die(f"Connexion MongoDB impossible : {e}")

    coll = client[DB_NAME][COLL_NAME]
    print(f"[INFO] Mongo OK | base={DB_NAME} collection={COLL_NAME}")

    docs = list(coll.find({}, PROJECTION))
    if not docs:
        die("Aucun document retourné.")

    df = pl.from_dicts(docs, infer_schema_length=None)

    # Typage
    df = df.with_columns([
        pl.col("last_scraped").cast(pl.Utf8, strict=False).str.strptime(pl.Date, strict=False),
        pl.col("availability_30").cast(pl.Int64, strict=False),
        pl.col("number_of_reviews").cast(pl.Int64, strict=False),
        pl.col("host_is_superhost").cast(pl.Utf8, strict=False),
        pl.col("room_type").cast(pl.Utf8, strict=False),
        pl.col("neighbourhood_cleansed").cast(pl.Utf8, strict=False),
    ])

    # Nettoyage données critiques
    df = df.drop_nulls([
        "last_scraped",
        "room_type",
        "availability_30",
        "neighbourhood_cleansed",
    ])

    df = df.filter(
        (pl.col("availability_30") >= 0) &
        (pl.col("availability_30") <= 30)
    )

    df = df.filter(
        (pl.col("neighbourhood_cleansed").str.len_chars() > 2) &
        (~pl.col("neighbourhood_cleansed").str.contains(r"^\[")) &
        (~pl.col("neighbourhood_cleansed").str.contains(r"^\d+$"))
    )

    # KPI
    df = df.with_columns([
        ((30 - pl.col("availability_30")) / 30).alias("taux_reservation_30j"),
        pl.col("last_scraped").dt.strftime("%Y-%m").alias("mois"),
    ])

    # Traduction room_type
    df = df.with_columns(
        pl.when(pl.col("room_type") == "Entire home/apt").then(pl.lit("Logement entier"))
        .when(pl.col("room_type") == "Private room").then(pl.lit("Chambre privée"))
        .when(pl.col("room_type") == "Shared room").then(pl.lit("Chambre partagée"))
        .when(pl.col("room_type") == "Hotel room").then(pl.lit("Chambre d’hôtel"))
        .otherwise(pl.col("room_type"))
        .alias("type_logement")
    ).drop("room_type")

    # 1
    (
        df.group_by(["mois", "type_logement"])
        .agg(pl.col("taux_reservation_30j").mean().alias("taux_reservation_moyen"))
        .sort(["mois", "type_logement"])
        .write_csv(f"{OUTDIR}/01_taux_reservation_moyen_par_mois_et_type_logement.csv")
    )

    # 2
    (
        df.select(pl.col("number_of_reviews").median().alias("mediane_nombre_avis"))
        .write_csv(f"{OUTDIR}/02_mediane_nombre_avis_tous_logements.csv")
    )

    # 3
    (
        df.with_columns(
            pl.when(pl.col("host_is_superhost") == "t")
            .then(pl.lit("Superhôte"))
            .otherwise(pl.lit("Non superhôte"))
            .alias("categorie_hote")
        )
        .group_by("categorie_hote")
        .agg(pl.col("number_of_reviews").median().alias("mediane_nombre_avis"))
        .write_csv(f"{OUTDIR}/03_mediane_nombre_avis_par_categorie_hote.csv")
    )

    # 4
    (
        df.group_by("neighbourhood_cleansed")
        .agg(pl.len().alias("nombre_annonces"))
        .sort("nombre_annonces", descending=True)
        .rename({"neighbourhood_cleansed": "quartier"})
        .write_csv(f"{OUTDIR}/04_densite_logements_par_quartier.csv")
    )

    # 5
    (
        df.group_by(["mois", "neighbourhood_cleansed"])
        .agg(pl.col("taux_reservation_30j").mean().alias("taux_reservation_moyen"))
        .with_columns(
            pl.col("taux_reservation_moyen")
            .rank(method="dense", descending=True)
            .over("mois")
            .alias("rang")
        )
        .filter(pl.col("rang") <= TOP_N)
        .rename({"neighbourhood_cleansed": "quartier"})
        .write_csv(f"{OUTDIR}/05_top_quartiers_taux_reservation_par_mois.csv")
    )

    print(f"[OK] Exports générés dans {OUTDIR}")


if __name__ == "__main__":
    main()
