"""
generate_dataset_metadata.py
-----------------------------
Generates data/dataset_metadata.json containing complete metadata specifications,
source attribution, licensing, schema descriptions, and dataset statistics for
the Kenya Multilingual Public Service Announcement (PSA) Translation Corpus.
"""

import sys, io, os, json
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA_DIR = "data"
MASTER_PSA = os.path.join(DATA_DIR, "Master_PSA_Only.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "dataset_metadata.json")

def generate():
    print("=" * 80)
    print("  GENERATING DATASET METADATA SPECIFICATION")
    print("=" * 80)

    total_rows = 0
    sw_count = 0
    guz_count = 0
    domains = {}

    if os.path.exists(MASTER_PSA):
        df = pd.read_csv(MASTER_PSA, dtype=str)
        total_rows = len(df)
        pending = "N/A - Pending Fine-Tuned Model Inference"
        sw_count = int(((df['Kiswahili'].fillna('') != '') & (df['Kiswahili'] != pending)).sum())
        guz_count = int(((df['Ekegusii'].fillna('') != '') & (df['Ekegusii'] != pending)).sum())
        domains = {str(k): int(v) for k, v in df['Domain'].value_counts().to_dict().items()}

    metadata = {
        "dataset_name": "Kenya Multilingual Public Service Announcement (PSA) Parallel Corpus",
        "version": "1.0.0",
        "release_date": "2026-07-27",
        "license": "Creative Commons Attribution 4.0 International (CC-BY-4.0)",
        "project_lead": "USIU-Africa NLP Research Group",
        "description": "A curated multilingual parallel corpus of verified public health, security, education, governance, and agricultural advisories in Kenya translated between English, Kiswahili, and Ekegusii (Gusii).",
        "language_pairs": {
            "English_Kiswahili": {
                "source_lang": "eng_Latn",
                "target_lang": "swh_Latn",
                "sentence_pairs": sw_count
            },
            "English_Ekegusii": {
                "source_lang": "eng_Latn",
                "target_lang": "guz_Latn",
                "sentence_pairs": guz_count
            }
        },
        "total_psa_sentences": total_rows,
        "domain_distribution": domains,
        "primary_data_sources": [
            {"source_id": "SRC-01", "name": "Ministry of Health Kenya (@MOH_Kenya)", "type": "Official Social Media / Portal", "domain": "Health"},
            {"source_id": "SRC-02", "name": "Ministry of Agriculture & Livestock Development", "type": "Government Advisories", "domain": "Agriculture"},
            {"source_id": "SRC-03", "name": "National Drought Management Authority (NDMA)", "type": "Early Warning PDF Bulletins", "domain": "Disaster/Health"},
            {"source_id": "SRC-04", "name": "Kenya National Commission on Human Rights (KNCHR)", "type": "Civic & Legal Bulletins", "domain": "Governance"},
            {"source_id": "SRC-05", "name": "Directorate of Criminal Investigations (DCI Kenya)", "type": "Security Alerts", "domain": "Security"},
            {"source_id": "SRC-06", "name": "Teachers Service Commission (TSC)", "type": "Official Recruitment & Bulletins", "domain": "Education"},
            {"source_id": "SRC-07", "name": "Kenya Institute of Curriculum Development (KICD)", "type": "Curriculum Announcements", "domain": "Education"},
            {"source_id": "SRC-08", "name": "National Police Service (NPS Kenya)", "type": "Community Safety Bulletins", "domain": "Security"},
            {"source_id": "SRC-09", "name": "IFRC / Kenya Red Cross Society", "type": "Humanitarian Relief Bulletins", "domain": "Health/Disaster"},
            {"source_id": "SRC-10", "name": "UNICEF Kenya", "type": "Maternal & Child Health Advisories", "domain": "Health"},
            {"source_id": "SRC-11", "name": "Kenya Plant Health Inspectorate Service (KEPHIS)", "type": "Biosecurity Notices", "domain": "Agriculture"},
            {"source_id": "SRC-12", "name": "Public Service Commission (PSC)", "type": "Government Notices", "domain": "Governance"}
        ],
        "schema_definition": [
            {"field": "PSA_ID", "type": "String", "description": "Unique record tracking identifier (e.g. PSA-HLT-0012)"},
            {"field": "Domain", "type": "String", "description": "Public advisory category (Health, Security, Education, Agriculture, Governance)"},
            {"field": "English", "type": "String", "description": "Normalized source public advisory sentence in English"},
            {"field": "Kiswahili", "type": "String", "description": "Standard Kiswahili translation"},
            {"field": "Ekegusii", "type": "String", "description": "Low-resource Ekegusii (Gusii) translation"},
            {"field": "PSA_Probability", "type": "Float", "description": "TF-IDF + Logistic Regression classification confidence score (0.0 to 1.0)"},
            {"field": "Is_PSA", "type": "Integer", "description": "Binary validation verdict (1 = Confirmed PSA, 0 = Non-PSA)"},
            {"field": "Source", "type": "String", "description": "Originating organization or portal"},
            {"field": "Date", "type": "String", "description": "Publication date or collection timestamp"}
        ]
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✔ Successfully saved metadata file: {OUTPUT_JSON}")

if __name__ == "__main__":
    generate()
