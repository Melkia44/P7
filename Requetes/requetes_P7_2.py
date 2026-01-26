import os
import sys
from pathlib import Path
from pymongo import MongoClient
import polars as pl


# -----------------------
# Paramètres projet
# -----------------------
DB_NAME = "P7MLO"
COLL_NAME = "listings"

OUTDIR = os.getenv("OUTDIR", "/app/outputs")
TOP_N = int(os.getenv("TOP_N", "5"))

# Si = "1", on supprime les CSV existants avant de réexporter
NETTOYER_OUTPUTS = os.getenv("NETTOYER_OUTPUTS", "0") == "1"

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
    print(f"[ERREUR] {msg}", file=sys.stderr)
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


def nettoyer_outputs(outdir: str):
    """Supprime les anciens CSV dans le dossier de sortie."""
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

    # Optionnel : nettoyage avant export (pratique quand on relance souvent)
    if NETTOYER_OUTPUTS:
        nettoyer_outputs(OUTDIR)

    uri = build_mongo_uri()

    # Connexion
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except Exception as e:
        die(f"Connexion MongoDB impossible (ping KO). Détail: {e}")

    coll = client[DB_NAME][COLL_NAME]
    print(f"[INFO] Mongo OK | base={DB_NAME} collection={COLL_NAME}")

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
        ((pl.lit(30) - pl.col("availability_30")) / pl.lit(30)).alias("taux_reservation_30j"),
        pl.col("last_scraped").dt.strftime("%Y-%m").alias("mois"),
    ])

    # Optionnel : traduction des types de logement 
    df = df.with_columns(
        pl.when(pl.col("room_type") == "Entire home/apt").then(pl.lit("Logement entier"))
        .when(pl.col("room_type") == "Private room").then(pl.lit("Chambre privée"))
        .when(pl.col("room_type") == "Shared room").then(pl.lit("Chambre partagée"))
        .when(pl.col("room_type") == "Hotel room").then(pl.lit("Chambre d’hôtel"))
        .otherwise(pl.col("room_type"))
        .alias("type_logement")
    ).drop("room_type")

    # 1) Taux de réservation moyen par mois et par type de logement
    t1 = (
        df.group_by(["mois", "type_logement"])
          .agg(pl.col("taux_reservation_30j").mean().alias("taux_reservation_moyen"))
          .sort(["mois", "type_logement"])
    )
    t1.write_csv(os.path.join(OUTDIR, "01_taux_reservation_moyen_par_mois_et_type_logement.csv"))

    # 2) Médiane du nombre d’avis (tous logements)
    (
        df.select(pl.col("number_of_reviews").median().alias("mediane_nombre_avis"))
          .write_csv(os.path.join(OUTDIR, "02_mediane_nombre_avis_tous_logements.csv"))
    )

    # 3) Médiane du nombre d’avis par catégorie d’hôte (superhost vs non)
    t3 = (
        df.with_columns(
            pl.when(pl.col("host_is_superhost") == "t")
              .then(pl.lit("Superhôte"))
              .otherwise(pl.lit("Non superhôte"))
              .alias("categorie_hote")
        )
        .group_by("categorie_hote")
        .agg(pl.col("number_of_reviews").median().alias("mediane_nombre_avis"))
        .sort("categorie_hote")
    )
    t3.write_csv(os.path.join(OUTDIR, "03_mediane_nombre_avis_par_categorie_hote.csv"))

    # 4) Densité de logements par quartier (volume d'annonces)
    t4 = (
        df.group_by("neighbourhood_cleansed")
          .agg(pl.len().alias("nombre_annonces"))
          .sort("nombre_annonces", descending=True)
          .rename({"neighbourhood_cleansed": "quartier"})
    )
    t4.write_csv(os.path.join(OUTDIR, "04_densite_logements_par_quartier.csv"))

    # 5) Quartiers avec le plus fort taux de réservation par mois (top N)
    t5 = (
        df.group_by(["mois", "neighbourhood_cleansed"])
          .agg(pl.col("taux_reservation_30j").mean().alias("taux_reservation_moyen"))
          .with_columns(
              pl.col("taux_reservation_moyen")
                .rank(method="dense", descending=True)
                .over("mois")
                .alias("rang")
          )
          .filter(pl.col("rang") <= TOP_N)
          .sort(["mois", "rang"])
          .rename({"neighbourhood_cleansed": "quartier"})
    )
    t5.write_csv(os.path.join(OUTDIR, "05_top_quartiers_taux_reservation_par_mois.csv"))

    print(f"[OK] Exports générés dans : {OUTDIR}")
    if NETTOYER_OUTPUTS:
        print("[INFO] Nettoyage préalable : anciens CSV supprimés.")


if __name__ == "__main__":
    main()
