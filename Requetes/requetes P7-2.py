import os
import sys
from pymongo import MongoClient
import polars as pl


# -----------------------
# Paramètres projet
# -----------------------
DB_NAME = "P7MLO"
COLL_NAME = "listings"

OUTDIR = os.getenv("OUTDIR", "outputs")
TOP_N = int(os.getenv("TOP_N", "5"))

# Connexion Mongo : on prend en priorité MONGO_URI.
# Sinon on construit l’URI à partir de MONGO_USER / MONGO_PASS.
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
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def build_mongo_uri() -> str:
    """
    Construit une URI MongoDB.
    - Si MONGO_URI est fourni, on l’utilise tel quel.
    - Sinon, on construit avec user/pass (si fournis), sans les afficher.
    """
    if MONGO_URI:
        return MONGO_URI

    # Sans auth (si Mongo n’en demande pas)
    if not MONGO_USER and not MONGO_PASS:
        return f"mongodb://{MONGO_HOST}:{MONGO_PORT}"

    # Avec auth (cas le plus courant)
    if not MONGO_USER or not MONGO_PASS:
        die("Il manque MONGO_USER ou MONGO_PASS (auth activée mais identifiants incomplets).")

    return f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/?authSource={MONGO_AUTH_SOURCE}"


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    uri = build_mongo_uri()
    # Connexion
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except Exception as e:
        die(f"Connexion MongoDB impossible (ping KO). Détail: {e}")

    coll = client[DB_NAME][COLL_NAME]
    print(f"[INFO] Mongo OK | db={DB_NAME} coll={COLL_NAME}")

    # Extraction
    docs = list(coll.find({}, PROJECTION))
    if not docs:
        die("Aucun document retourné. Vérifie DB/collection/champs.")

    df = pl.from_dicts(docs)

    # Typage / nettoyage minimal
    df = df.with_columns([
        pl.col("last_scraped").cast(pl.Utf8, strict=False).str.strptime(pl.Date, strict=False),
        pl.col("availability_30").cast(pl.Int64, strict=False),
        pl.col("number_of_reviews").cast(pl.Int64, strict=False),
        pl.col("host_is_superhost").cast(pl.Utf8, strict=False),
        pl.col("room_type").cast(pl.Utf8, strict=False),
        pl.col("neighbourhood_cleansed").cast(pl.Utf8, strict=False),
    ]).drop_nulls(["last_scraped", "room_type", "availability_30", "neighbourhood_cleansed"])

    # KPI réservation (proxy) : (30 - dispo_30)/30
    df = df.with_columns([
        ((pl.lit(30) - pl.col("availability_30")) / pl.lit(30)).alias("booking_rate_30d"),
        pl.col("last_scraped").dt.strftime("%Y-%m").alias("month"),
    ])

    # 1) Taux de réservation moyen par mois et par type de logement
    t1 = (
        df.group_by(["month", "room_type"])
          .agg(pl.col("booking_rate_30d").mean().alias("avg_booking_rate"))
          .sort(["month", "room_type"])
    )
    t1.write_csv(os.path.join(OUTDIR, "01_booking_rate_by_month_room_type.csv"))

    # 2) Médiane du nombre d’avis (tous logements)
    df.select(pl.col("number_of_reviews").median().alias("median_reviews_all")) \
      .write_csv(os.path.join(OUTDIR, "02_median_reviews_all.csv"))

    # 3) Médiane du nombre d’avis par catégorie d’hôte (superhost vs non)
    t3 = (
        df.with_columns(
            pl.when(pl.col("host_is_superhost") == "t")
              .then(pl.lit("superhost"))
              .otherwise(pl.lit("non_superhost"))
              .alias("host_category")
        )
        .group_by("host_category")
        .agg(pl.col("number_of_reviews").median().alias("median_reviews"))
        .sort("host_category")
    )
    t3.write_csv(os.path.join(OUTDIR, "03_median_reviews_by_host_category.csv"))

    # 4) Densité de logements par quartier (volume d'annonces)
    t4 = (
        df.group_by("neighbourhood_cleansed")
          .agg(pl.len().alias("listings_count"))
          .sort("listings_count", descending=True)
          .rename({"neighbourhood_cleansed": "neighbourhood"})
    )
    t4.write_csv(os.path.join(OUTDIR, "04_listings_density_by_neighbourhood.csv"))

    # 5) Quartiers avec le plus fort taux de réservation par mois (top N)
    t5 = (
        df.group_by(["month", "neighbourhood_cleansed"])
          .agg(pl.col("booking_rate_30d").mean().alias("avg_booking_rate"))
          .with_columns(
              pl.col("avg_booking_rate")
                .rank(method="dense", descending=True)
                .over("month")
                .alias("rank")
          )
          .filter(pl.col("rank") <= TOP_N)
          .sort(["month", "rank"])
          .rename({"neighbourhood_cleansed": "neighbourhood"})
    )
    t5.write_csv(os.path.join(OUTDIR, "05_top_neighbourhoods_by_booking_rate_per_month.csv"))

    print(f"[OK] Exports générés dans: {OUTDIR}")


if __name__ == "__main__":
    main()
