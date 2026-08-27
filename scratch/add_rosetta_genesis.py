import os
import pandas as pd
import numpy as np
import io

print("=== Adding Rosetta Project Genesis 1 Rows & Running Audit Pipeline ===")

new_rows = [
    {
        "id": 1,
        "ekegusii": "Agwo Omochakano Nyasae nigo atongete igoro na ense.",
        "kiswahili": "",
        "english": "In the beginning God created the heaven and the earth.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 2,
        "ekegusii": "Na ense nigo yarenge bosa, tiyarenge na kieni kende, nigo yaare nomosunte otubete endiba; na Omoika o Nyasae nigo orenge goetanana igoro y'amache.",
        "kiswahili": "",
        "english": "And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 3,
        "ekegusii": 'Nyasae agateba, "Tiga oborabu bobeo", na oborabu bokabao.',
        "kiswahili": "",
        "english": "And God said, Let there be light: and there was light.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 4,
        "ekegusii": "Nyasae akarora oborabu obwo ng'a nobuya, agaatanana oborabu korwa ase omosunte.",
        "kiswahili": "",
        "english": "And God saw the light, that it was good: and God divided the light from the darkness.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 5,
        "ekegusii": "Nyasae akaroka oborabu boria omobaso, na omosunte akayoroka obotuko. Erio agwo rikaba mogoroba, naende bokaba mambia, rituko rie ritang'ani.",
        "kiswahili": "",
        "english": "And God called the light Day, and the darkness he called Night. And the evening and the morning were the first day.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 6,
        "ekegusii": 'Nyasae agateba, "Tiga eaanga ebe egati-gati y\'amache, yaatanane amache korwa ase amache ande."',
        "kiswahili": "",
        "english": "And God said, Let there be a firmament in the midst of the waters, and let it divide the waters from the waters.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 7,
        "ekegusii": "Igo Nyasae agakora eaanga eria, agaatanana amache ayare inse y'eaanga korwa ase amache ayare igoro y'eaanga; akaba boigo.",
        "kiswahili": "",
        "english": "And God made the firmament, and divided the waters which were under the firmament from the waters which were above the firmament: and it was so.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 8,
        "ekegusii": "Nyasae akaroka eaanga eria igoro. Erio agwo rikaba mogoroba, naende bokaba mambia, rituko ria kabere.",
        "kiswahili": "",
        "english": "And God called the firmament Heaven. And the evening and the morning were the second day.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 9,
        "ekegusii": 'Erio Nyasae agateba, "Tiga amache ayare inse ya igoro asangererekane aase aamo, ense enyomo erorekane"; akaba boigo.',
        "kiswahili": "",
        "english": "And God said, Let the waters under the heaven be gathered together unto one place, and let the dry land appear: and it was so.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 10,
        "ekegusii": "Nyasae akaroka eria enyomo ense, na amache aria asangererekanete amo akayaroka chinyancha. Nyasae akarora ayio ng'a namaya.",
        "kiswahili": "",
        "english": "And God called the dry land Earth; and the gathering together of the waters called he Seas: and God saw that it was good.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 11,
        "ekegusii": 'Naende Nyasae agateba, "Tiga ense emere ebinto bigokina, buna ebimeri bikorua chimbusuro, na emete ekwama amatunda are ne chimbusuro chire imeo, kera oyomo ase egesaku kiaye ekenyene"; akaba boigo.',
        "kiswahili": "",
        "english": "And God said, Let the earth bring forth grass, the herb yielding seed, and the fruit tree yielding fruit after his kind, whose seed is in itself, upon the earth: and it was so.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 12,
        "ekegusii": "Na ense ekamera bionsi ebigokina: ebimeri bikorua chimbusuro chiabo chinyene, na emete ekwama amatunda are ne chimbusuro chire imeo, kera oyomo ase egesaku kiaye ekenyene. Nyasae akarora ayio ng'a namaya.",
        "kiswahili": "",
        "english": "And the earth brought forth grass, and herb yielding seed after his kind, and the tree yielding fruit, whose seed was in itself, after his kind: and God saw that it was good.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 13,
        "ekegusii": "Erio agwo rikaba mogoroba, naende bokaba mambia, rituko ria gatato.",
        "kiswahili": "",
        "english": "And the evening and the morning were the third day.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 14,
        "ekegusii": 'Erio Nyasae agateba, "Tiga emebaso ebe ase eaanga aaria igoro, yaatanane omobaso korwa ase obotuko; ero ebe ebimanyererio bikworokia Amatuko Amanene, na amatuko, na emiaka,"',
        "kiswahili": "",
        "english": "And God said, Let there be lights in the firmament of the heaven to divide the day from the night; and let them be for signs, and for seasons, and for days, and years:",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 15,
        "ekegusii": "ebe emebaso ase eaanga aaria igoro, eyee ense omobaso na oborabu; akaba boigo.",
        "kiswahili": "",
        "english": "And let them be for lights in the firmament of the heaven to give light upon the earth: and it was so.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 16,
        "ekegusii": "Nyasae agakora emebaso emenene ebere. Omobaso oria omonene ase eyio ebere ogambere ekero kia mobaso, na oria omoke ekero kia botuko; agakora ne ching'enang'eni boigo.",
        "kiswahili": "",
        "english": "And God made two great lights; the greater light to rule the day, and the lesser light to rule the night: he made the stars also.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 17,
        "ekegusii": "Erio Nyasae akayebeka ase eaanga aaria igoro, eyee ense omobaso na oborabu,",
        "kiswahili": "",
        "english": "And God set them in the firmament of the heaven to give light upon the earth,",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 18,
        "ekegusii": "egambere omobaso na obotuko, na gwatanana oborabu korwa ase omosunte. Nyasae akarora ayio ng'a namaya.",
        "kiswahili": "",
        "english": "And to rule over the day and over the night, and to divide the light from the darkness: and God saw that it was good.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 19,
        "ekegusii": "Erio agwo rikaba mogoroba, naende bokaba mambia, rituko ria kane.",
        "kiswahili": "",
        "english": "And the evening and the morning were the fourth day.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 20,
        "ekegusii": 'Nyasae agateba, "Tiga amache atware obonge bw\'ebitongwa bire moyo, ne chinyoni chikoiruruka igoro ase ense, chigoetanana inse y\'eaanga ya igoro."',
        "kiswahili": "",
        "english": "And God said, Let the waters bring forth abundantly the moving creature that hath life, and fowl that may fly above the earth in the open firmament of heaven.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 21,
        "ekegusii": "Naende Nyasae agatonga ebitongwa ebinene bire ase amache ime, amo nebitongwa binde bionsi bire moyo, ebio bigoetanana ase obonge imeo; agatonga kera egesaku nengencho yaye. Agatonga ne chinyoni chibwate chimbaba, kera egesaku nengencho yaye. Nyasae akarora ayio ng'a namaya.",
        "kiswahili": "",
        "english": "And God created great whales, and every living creature that moveth, which the waters brought forth abundantly, after their kind, and every winged fowl after his kind: and God saw that it was good.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 22,
        "ekegusii": 'Erio agwo Nyasae akabisesenia, agateba, "Ibora, momenteke, moichore ime ase chinyancha, ne chinyoni chibuche ase ense."',
        "kiswahili": "",
        "english": "And God blessed them, saying, Be fruitful, and multiply, and fill the waters in the seas, and let fowl multiply in the earth.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 23,
        "ekegusii": "Erio agwo rikaba mogoroba, naende bokaba mambia, rituko ria gatano.",
        "kiswahili": "",
        "english": "And the evening and the morning were the fifth day.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 24,
        "ekegusii": 'Naende Nyasae agateba, "Tiga ense etware ebitongwa bire moyo, kera egesaku nengencho yaye: ching\'iti chigotugwa, na echikwagura, ne ching\'iti chi\'orosana, kera egesaku nengencho yaye"; akaba boigo.',
        "kiswahili": "",
        "english": "And God said, Let the earth bring forth the living creature after his kind, cattle, and creeping thing, and beast of the earth after his kind: and it was so.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 25,
        "ekegusii": "Nyasae agakora ching'iti chi'orosana, kera egesaku nengencho yaye, ne ching'iti chigotugwa, kera egesaku nengencho yaye, ne ching'iti chikwagura, kera egesaku nengencho yaye. Nyasae akarora ayio ng'a namaya.",
        "kiswahili": "",
        "english": "And God made the beast of the earth after his kind, and cattle after their kind, and every thing that creepeth upon the earth after his kind: and God saw that it was good.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 26,
        "ekegusii": 'Erio Nyasae agateba, "Tiga tokore omonto ase omogwekano oito, otogwekaine, agambere chinswe chia nyancha, ne chinyoni chikoiruruka igoro, ne ching\'iti chigotugwa, ne ching\'iti chionsi chi\'orosana, na ense yonsi, ne ching\'iti chionsi chikwagura igoro ase ense."',
        "kiswahili": "",
        "english": "And God said, Let us make man in our image, after our likeness: and let them have dominion over the fish of the sea, and over the fowl of the air, and over the cattle, and over all the earth, and over every creeping thing that creepeth upon the earth.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 27,
        "ekegusii": "Nabo Nyasae agatonga mwanyabanto ase omogwekano oye omonyene; ase omogwekano o Nyasae ere akamotonga. Agatonga omosacha na omokungu.",
        "kiswahili": "",
        "english": "So God created man in his own image, in the image of God created he him; male and female created he them.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 28,
        "ekegusii": 'Nyasae akabasesenia, akabatebia, "Moibore, momenteke, moichore ase ense, na moyegambere. Mogambere ne chinswe chia nyancha, ne chinyoni chikoiruruka, na kera egetongwa kere moyo kegotaara igoro ase ense,"',
        "kiswahili": "",
        "english": "And God blessed them, and God said unto them, Be fruitful, and multiply, and replenish the earth, and subdue it: and have dominion over the fish of the sea, and over the fowl of the air, and over every living thing that moveth upon the earth.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 29,
        "ekegusii": 'Naende Nyasae agateba, "Rora, nabaeire kera ekemeri gekwama chimbusuro ase ense engima, na kera omote okwama amatunda are ne chimbusuro chire imeo; nabio birabe endagera yaino."',
        "kiswahili": "",
        "english": "And God said, Behold, I have given you every herb bearing seed, which is upon the face of all the earth, and every tree, in the which is the fruit of a tree yielding seed; to you it shall be for meat.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 30,
        "ekegusii": 'Na ebimeri ebibese bionsi nabirure bibe endagera ye ching\'iti chionsi chire moyo ase ense"; akaba boigo.',
        "kiswahili": "",
        "english": "And to every beast of the earth, and to every fowl of the air, and to every thing that creepeth upon the earth, wherein there is life, I have given every green herb for meat: and it was so.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    },
    {
        "id": 31,
        "ekegusii": "Nyasae akarigereria ebinto ebio bionsi akorete, na birobio nigo biarenge ebiya mono. Erio agwo rikaba mogoroba, naende bokaba mambia, rituko ria gatano na rimo.",
        "kiswahili": "",
        "english": "And God saw every thing that he had made, and, behold, it was very good. And the evening and the morning were the sixth day.",
        "source": "Ebibilia Enchenu (Bible Society of Kenya, 1990) via Rosetta Project",
        "source_url": "https://archive.org/details/rosettaproject_guz_gen-1",
        "source_type": "Religious text (scripture, OCR scan; English = standard public-domain KJV aligned by verse)"
    }
]

df_new = pd.DataFrame(new_rows)
print(f"Parsed {len(df_new)} new Rosetta Genesis 1 rows.")

# Load existing dataset
master_path = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\dataset\ekegusii_master_dataset.csv"
df_master = pd.read_csv(master_path)
print(f"Existing Master Dataset count: {len(df_master)}")

# Append & Deduplicate
df_combined = pd.concat([df_master, df_new], ignore_index=True)
df_combined.drop_duplicates(subset=["ekegusii", "english"], inplace=True)
df_combined["id"] = range(1, len(df_combined) + 1)
print(f"Combined & Deduplicated count: {len(df_combined)}")

# Save to dataset/ and data/master_corpus/
df_combined.to_csv(master_path, index=False, encoding="utf-8-sig")
os.makedirs(r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\master_corpus", exist_ok=True)
df_combined.to_csv(r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\master_corpus\ekegusii_master_dataset.csv", index=False, encoding="utf-8-sig")

dataset_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\dataset"
splits_dir = os.path.join(dataset_dir, "splits")

# Source Statistics
src_stats = df_combined.groupby("source").agg(
    total_rows=("id", "count"),
    english_pairs=("english", lambda x: x.notna().sum()),
    swahili_pairs=("kiswahili", lambda x: (x.notna() & (x.str.strip() != "")).sum()),
    source_types=("source_type", lambda x: x.iloc[0] if len(x)>0 else "")
).reset_index()
src_stats.to_csv(os.path.join(dataset_dir, "source_statistics.csv"), index=False)

# Duplicates check
dup_mask = df_combined.duplicated(subset=["ekegusii", "english"], keep=False)
df_duplicates = df_combined[dup_mask].copy()
df_duplicates.to_csv(os.path.join(dataset_dir, "ekegusii_duplicates.csv"), index=False)

# Suspect Check
def check_suspect(row):
    guz = str(row["ekegusii"]) if pd.notna(row["ekegusii"]) else ""
    eng = str(row["english"]) if pd.notna(row["english"]) else ""
    reasons = []
    if len(guz) < 4: reasons.append("too_short_ekegusii")
    if eng and len(eng) < 4: reasons.append("too_short_english")
    if eng:
        ratio = len(guz) / (len(eng) + 1e-5)
        if ratio > 4.5 or ratio < 0.2: reasons.append(f"length_ratio_mismatch_{ratio:.2f}")
    return "; ".join(reasons) if reasons else "clean"

df_combined["audit_flag"] = df_combined.apply(check_suspect, axis=1)
df_suspect = df_combined[df_combined["audit_flag"] != "clean"].copy()
df_suspect.to_csv(os.path.join(dataset_dir, "ekegusii_suspect_rows.csv"), index=False)

# Clean Dataset
df_clean = df_combined.drop_duplicates(subset=["ekegusii", "english"]).copy()
df_clean = df_clean[df_clean["audit_flag"] == "clean"].copy()
df_clean.drop(columns=["audit_flag"], inplace=True)
df_clean.to_csv(os.path.join(dataset_dir, "ekegusii_clean.csv"), index=False)

# Audit Report
audit_report = pd.DataFrame([{
    "metric": "Total Master Corpus Rows", "value": len(df_combined)
}, {
    "metric": "Unique Sentence Pairs", "value": len(df_combined.drop_duplicates(subset=["ekegusii", "english"]))
}, {
    "metric": "Duplicate Rows", "value": len(df_duplicates)
}, {
    "metric": "Suspect Mismatch Rows", "value": len(df_suspect)
}, {
    "metric": "Clean Production Rows", "value": len(df_clean)
}])
audit_report.to_csv(os.path.join(dataset_dir, "ekegusii_audit_report.csv"), index=False)

# Splits (80/10/10)
np.random.seed(42)
shuffled_df = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
n_total = len(shuffled_df)
n_train = int(n_total * 0.80)
n_val = int(n_total * 0.10)

df_train = shuffled_df.iloc[:n_train].copy()
df_val = shuffled_df.iloc[n_train:n_train+n_val].copy()
df_test = shuffled_df.iloc[n_train+n_val:].copy()

df_train.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
df_val.to_csv(os.path.join(splits_dir, "validation.csv"), index=False)
df_test.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

print(f"=== UPDATED DATASET STATS ===")
print(f"Total Master Corpus Rows: {len(df_combined)}")
print(f"Clean Production Rows: {len(df_clean)}")
print(f"Train split: {len(df_train)} | Val split: {len(df_val)} | Test split: {len(df_test)}")
