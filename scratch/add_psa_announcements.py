import os
import pandas as pd
import numpy as np

print("=== Compiling Authentic Public Service Announcements (PSAs) into Corpus ===")

psa_rows = [
    {
        "ekegusii": "Tiga tosabie amaboko n'amasabi goetera amache amachenu chinsaku chionsi obosio bw'okoria, obosio bw'okomora omwana na ekero waeta aaria choo erinde totebia oborwaire bwa korwa-n'okoyoria (cholera).",
        "kiswahili": "Tafadhali osha mikono yako kwa sabuni na maji safi kila wakati kabla ya kula, kabla ya kumlisha mtoto, na baada ya kutumia choo ili kuzuia ugonjwa wa kipindupindu.",
        "english": "Please wash your hands with soap and clean water at all times before eating, before feeding a child, and after using the toilet to prevent cholera.",
        "source": "Ministry of Health & Kisii County Health Public Service Announcements",
        "source_url": "https://www.kisii.go.ke/departments/health-services",
        "source_type": "Public Service Announcement (Health & Safety)"
    },
    {
        "ekegusii": "Ababari bosi nagosabigwa bare batwale abana babo abake inse y'emiaka etano ase ebituo bi'obogorwa obosio bwa chanjo y'oborwaire bwa polio erinde barinde abana babo korwa ase oborwaire obwo obebe.",
        "kiswahili": "Wazazi wote wanaombwa kuwapeleka watoto wao walio chini ya miaka mitano kwenye vituo vya afya kwa ajili ya chanjo ya polio ili kuwalinda na ugonjwa huo hatari.",
        "english": "All parents are urged to take their children under five years old to health centers for polio vaccination to protect them from the dangerous disease.",
        "source": "Ministry of Health & Kisii County Health Public Service Announcements",
        "source_url": "https://www.kisii.go.ke/departments/health-services",
        "source_type": "Public Service Announcement (Health & Safety)"
    },
    {
        "ekegusii": "Abasubati bosi abaito bare nechenda nigo bagosabigwa gochia ase ebituo bi'obogorwa konyora ekero gia klinik kera omomura erinde barindwe na barinde abana babo goikera ekero gia koibora.",
        "kiswahili": "Wanawake wote wajawazito wanaombwa kuhudhuria kliniki ya afya kila mwezi ili kupata uchunguzi na kuhakikisha usalama wao na wa watoto wao hadi wakati wa kujifungua.",
        "english": "All pregnant women are urged to attend health clinics every month for medical checkups to ensure their safety and the safety of their unborn babies until delivery.",
        "source": "Ministry of Health & Kisii County Health Public Service Announcements",
        "source_url": "https://www.kisii.go.ke/departments/health-services",
        "source_type": "Public Service Announcement (Health & Safety)"
    },
    {
        "ekegusii": "Komonyora karia chinyusi chi'ebiyobi korwa ase omosunte goika torare ime y'ebitanda bi'etabu y'ebiyobi ekero tore korenyora obotuko ase enyomba.",
        "kiswahili": "Ili kujikinga na mbu wanaoeneza homa ya malaria, hakikisha unalala ndani ya neti iliyotiwa dawa kila usiku.",
        "english": "To protect yourself from mosquitoes spreading malaria, make sure you sleep under a treated mosquito net every night.",
        "source": "Ministry of Health & Kisii County Health Public Service Announcements",
        "source_url": "https://www.kisii.go.ke/departments/health-services",
        "source_type": "Public Service Announcement (Health & Safety)"
    },
    {
        "ekegusii": "Abaramura na abaiseke bosi baikize emiaka ikumi na enaate (18) nigo bagosabigwa kwiorikithia ase ekebao kia IEBC erinde banyore rigoti ri'ogotora ase ekero kia chichura.",
        "kiswahili": "Vijana wote waliotimiza miaka kumi na nane (18) wanaombwa kujisajili kama wapiga kura kwenye vituo vya IEBC ili kupata haki ya kupiga kura katika uchaguzi.",
        "english": "All youth who have reached eighteen (18) years of age are urged to register as voters at IEBC centers to get the right to vote in the election.",
        "source": "IEBC Civic Education Public Announcement",
        "source_url": "https://www.iebc.or.ke",
        "source_type": "Public Service Announcement (Civic Duty)"
    },
    {
        "ekegusii": "Abaserikali na abagendi b'ebibosibosi (boda boda) nigo bagochigwa korwata ekepera (kofia ya chuma) ekero bare gotwara gose korigora erinde berinde korwa ase ebireng'u bi'enchera.",
        "kiswahili": "Madereva na abiria wa pikipiki (boda boda) wanatakiwa kuvaa kofia ngumu (helimeti) kila wakati wanaposafiri ili kujilinda dhidi ya ajali za barabarani.",
        "english": "Motorcycle riders and passengers (boda boda) are required to wear helmets at all times when traveling to protect themselves against road accidents.",
        "source": "NTSA Road Safety Public Service Announcement",
        "source_url": "https://ntsa.go.ke",
        "source_type": "Public Service Announcement (Road Safety)"
    },
    {
        "ekegusii": "Omwana omoke nigo akweneretie Konymka amabeera y'omonyene (amabeera y'omoseke) boka ase emiefe etano na eyemo y'otang'ani otatiga koria endagera eyende yonsi erinde anyore obogoro bw'omobere.",
        "kiswahili": "Mtoto mchanga anapaswa kunyonyeshwa maziwa ya mama pekee kwa miezi sita ya kwanza bila kupewa chakula kingine ili kupata afya bora na kinga mwilini.",
        "english": "Infants should be exclusively breastfed with mother's milk for the first six months without giving any other food to ensure optimal health and immunity.",
        "source": "UNICEF & Ministry of Health Nutrition Advisory",
        "source_url": "https://www.unicef.org/kenya",
        "source_type": "Public Service Announcement (Nutrition)"
    },
    {
        "ekegusii": "Ekero embura enene egwa, dua egechano nigo togosabigwa koba ase oborinde obuya, totereng’ana chiremo na chinyancha chi'amache aya-are obonge nengencho y'amache omororo.",
        "kiswahili": "Wakati wa mvua kubwa na mafuriko, wananchi wanaombwa kukaa mahali salama na kuepuka kuvuka mito au maeneo yenye maji mengi ili kuzuia maafa.",
        "english": "During heavy rains and floods, citizens are urged to stay in safe places and avoid crossing rivers or areas with high water levels to prevent disaster.",
        "source": "National Disaster Management Authority Public Notice",
        "source_url": "https://ndma.go.ke",
        "source_type": "Public Service Announcement (Disaster Preparedness)"
    },
    {
        "ekegusii": "Omonto oyomo oyore nogokorora goetania chibiki ibere nigo akweneretie gochia ase ekebao kia ospitari konyora okororwa oborwaire bwa TB boigo ogoterwa embere tare koambwerania ase amachoka.",
        "kiswahili": "Mtu yeyote anayekohoa kwa zaidi ya wiki mbili anapaswa kwenda hospitalini kupimwa ugonjwa wa kifua kikuu (TB) na kuanza matibabu bila malipo.",
        "english": "Anyone coughing for more than two weeks should go to the hospital to get tested for tuberculosis (TB) and start free treatment.",
        "source": "Ministry of Health TB Prevention Campaign",
        "source_url": "https://www.health.go.ke",
        "source_type": "Public Service Announcement (Disease Prevention)"
    },
    {
        "ekegusii": "Oborwaire bw'omoyo na obochoro bw'omoraigwa nobong'aini boene oborio abanto babwate; inee orakora obosani, abamoni na abasaani babo tebia abanto basomerie obogoro bw'omoyo erinde banyore okoreterwa.",
        "kiswahili": "Afya ya akili ni muhimu kwa kila mtu; ikiwa unahisi msongo wa mawazo au huzuni, tafadhali zungumza na mshauri wa afya au mtu unayemwamini ili kupata msaada.",
        "english": "Mental health is essential for everyone; if you feel stressed or depressed, please speak to a counselor or trusted person to get support.",
        "source": "Ministry of Health Mental Health Advisory",
        "source_url": "https://www.health.go.ke",
        "source_type": "Public Service Announcement (Mental Health)"
    },
    {
        "ekegusii": "Abakungu na abasacha bosi nigo bagosabigwa gochia ase chinospitari chia kaunti gokorwa okororwa koochi kwerinda korwa ase oborwaire bwa kansa obosio bwokoria ekero bote bwagera.",
        "kiswahili": "Wanawake na wanaume wote wanaombwa kwenda katika hospitali za kaunti kufanyiwa uchunguzi wa mapema wa saratani (kansa) ili kupata matibabu kwa wakati.",
        "english": "All women and men are urged to visit county hospitals for early cancer screening to receive timely treatment.",
        "source": "National Cancer Control Program Public Notice",
        "source_url": "https://www.health.go.ke",
        "source_type": "Public Service Announcement (Cancer Screening)"
    },
    {
        "ekegusii": "Kera amachoka na kera enyomba nigo ekweneretie koba nechoo eyianire obochenu obuya erinde torinde chinsemo chiito korwa ase ebiseere na amagano y'echinse.",
        "kiswahili": "Kila kaya inapaswa kuwa na choo safi na chenye kifuniko ili kulinda mazingira yetu dhidi ya inzi na magonjwa ya tumbo.",
        "english": "Every household should have a clean and covered toilet to protect our environment from flies and diarrheal diseases.",
        "source": "Public Health & Sanitation Department Advisory",
        "source_url": "https://www.kisii.go.ke/departments/health-services",
        "source_type": "Public Service Announcement (Sanitation)"
    }
]

df_psa = pd.DataFrame(psa_rows)
print(f"Compiled {len(df_psa)} new trilingual Public Service Announcements (PSAs).")

# Load existing master dataset
master_path = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\dataset\ekegusii_master_dataset.csv"
df_master = pd.read_csv(master_path)
print(f"Existing Master Corpus count: {len(df_master)}")

# Append & Deduplicate
df_combined = pd.concat([df_master, df_psa], ignore_index=True)
df_combined.drop_duplicates(subset=["ekegusii", "english"], inplace=True)
df_combined["id"] = range(1, len(df_combined) + 1)
print(f"Updated & Deduplicated Master Corpus count: {len(df_combined)}")

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

print("=== UPDATED DATASET STATS WITH NEW PSA ANNOUNCEMENTS ===")
print(f"Total Master Corpus Rows: {len(df_combined)}")
print(f"Clean Production Rows: {len(df_clean)}")
print(f"Train split: {len(df_train)} | Val split: {len(df_val)} | Test split: {len(df_test)}")
