import os
import pandas as pd
import numpy as np

print("=== Expanding Public Service Announcement (PSA) Dataset ===")

additional_psa_rows = [
    {
        "id": 13,
        "ekegusii": "Gocha koinyora ribaga ri’okororwa oborwaire bwa HIV/AIDS tare kobera chinsoni; chia ase ebituo bi’obogorwa konyora okororwa na obokoreri obwa boene boebeyanere n’amasabi.",
        "kiswahili": "Kwenda kupimwa virusi vya UKIMWI (HIV) si aibu; tembelea vituo vya afya vilivyo karibu ili kupimwa na kupata ushauri nasaha na matibabu bila malipo.",
        "english": "Going to get tested for HIV/AIDS is not shameful; visit nearby health centers to get tested and receive free counseling and treatment.",
        "source": "NASCOP & Ministry of Health HIV Prevention Campaign",
        "source_url": "https://www.nascop.or.ke",
        "source_type": "Public Service Announcement (Health & Safety)"
    },
    {
        "id": 14,
        "ekegusii": "Abana bosi b'amachora b’amang'ana asatu bagasabigwa koria emeti y’okwita chinzoka ase amachora yabo obosio bwa wiki eyo echete erinde barinde obogoro bw’emibere yabo.",
        "kiswahili": "Wanafunzi wote katika shule za msingi watapewa dawa za kuua minyoo shuleni wiki ijayo ili kulinda afya na maendeleo yao ya kimasomo.",
        "english": "All primary school pupils will be given deworming tablets at school next week to protect their health and educational growth.",
        "source": "Ministry of Education & MOH National Deworming Program",
        "source_url": "https://www.education.go.ke",
        "source_type": "Public Service Announcement (Child Health)"
    },
    {
        "id": 15,
        "ekegusii": "Tiga tobire amache y’okunywa gose tokore emeti y’okwira obori ase amache korwa ase egechano geria toranogwate oborwaire bwa Taifodi (typhoid).",
        "kiswahili": "Hakikisha unachemsha maji ya kunywa au kutumia dawa ya kutasa maji ili kujikinga na ugonjwa wa homa ya matumbo (typhoid).",
        "english": "Ensure you boil drinking water or use water treatment chemicals to protect yourself against typhoid fever.",
        "source": "Public Health Sanitation Advisory",
        "source_url": "https://www.kisii.go.ke/departments/health-services",
        "source_type": "Public Service Announcement (Water & Hygiene)"
    },
    {
        "id": 16,
        "ekegusii": "Abaparami bosi ase ekaunti ya Kisii na Nyamira bagosabigwa kwiorikithia ase chi-opisi chia makuru erinde banyore okoria chimbosio chi’emeboso chi’ing'okwe korwa ase serikali.",
        "kiswahili": "Wakulima wote katika kaunti za Kisii na Nyamira wanaombwa kujisajili kwa maofisa wa kilimo ili kupata mbolea ya ruzuku kutoka kwa serikali.",
        "english": "All farmers in Kisii and Nyamira counties are requested to register with agriculture officers to access subsidized fertilizer from the government.",
        "source": "Ministry of Agriculture Subsidized Fertilizer Program",
        "source_url": "https://kilimo.go.ke",
        "source_type": "Public Service Announcement (Agriculture)"
    },
    {
        "id": 17,
        "ekegusii": "Abaparami b'ebiaba (chimbaso) bagosabigwa kororera amabi ebiaba biabo na gosika emeti eyeianire ekero barora ebinyonyi bi'okuria ebiremo (armyworms) bwa-achere.",
        "kiswahili": "Wakulima wa mahindi wanaombwa kukagua mashamba yao mara kwa mara na kupiga dawa sahihi mara wanapoona mabuu ya viwavi jeshi (armyworms).",
        "english": "Maize farmers are urged to inspect their fields regularly and spray recommended pesticides upon detecting fall armyworm larvae.",
        "source": "KALRO & Agriculture Pest Advisory",
        "source_url": "https://kalro.org",
        "source_type": "Public Service Announcement (Pest Control)"
    },
    {
        "id": 18,
        "ekegusii": "Takeragana na omonto oyomo oyora-okoborrie PIN yao y’omobira gose emechando y’aba-benta; Safaricom tesabana PIN yao ekero bagokoboria ase omobira.",
        "kiswahili": "Usimpe mtu yeyote nambari yako ya siri (PIN) ya M-Pesa au maelezo yako ya benki; Safaricom haitawahi kuomba PIN yako kupitia simu.",
        "english": "Never share your M-Pesa secret PIN or banking details with anyone; Safaricom will never ask for your PIN over the phone.",
        "source": "Safaricom & Consumer Protection Cyber Awareness",
        "source_url": "https://www.safaricom.co.ke",
        "source_type": "Public Service Announcement (Cyber Security)"
    },
    {
        "id": 19,
        "ekegusii": "Amasikani na chimbara chia ng'ano tetwa goikwa ase ekaunti eyito; inee norabwate okoborana gose okohinya torika omobira bwa eke-nini obosio bwa 1195 konyora okoreterwa.",
        "kiswahili": "Ukatili wa kijinsia na dhuluma dhidi ya wanawake au wanaume haukubaliki; ukishuhudia au kufanyiwa kitendo hicho piga simu ya bure 1195 kupata msaada.",
        "english": "Gender-based violence and abuse are unacceptable; if you experience or witness violence, call the toll-free number 1195 for help.",
        "source": "State Department for Gender & GBV Helpline 1195",
        "source_url": "https://gender.go.ke",
        "source_type": "Public Service Announcement (Social Protection)"
    },
    {
        "id": 20,
        "ekegusii": "Abamura na abaiseke basabere chikitabu chia kororera omobere (National ID) nigo bagosabigwa gochia ase chi-opisi chia Kamisana ya Sub-County (Deputy County Commissioner) kwaitwa chikitabu chiabo.",
        "kiswahili": "Wananchi wote walioomba vitambulisho vya kitaifa (ID) wanaombwa kwenda katika ofisi za manaibu wa kamishna wa wilaya kuchukua vitambulisho vyao.",
        "english": "All citizens who applied for National Identification Cards (IDs) are requested to visit Sub-County Commissioner offices to collect their cards.",
        "source": "National Registration Bureau Public Announcement",
        "source_url": "https://immigration.go.ke",
        "source_type": "Public Service Announcement (Civic Affairs)"
    },
    {
        "id": 21,
        "ekegusii": "Toronde chinsemo chiito na kosima emete eyeno obosio bw’ekero gia embura erinde torinde amache na amachoka korwa ase enyome (dry seasons).",
        "kiswahili": "Tupeperushe bendera ya kutunza mazingira kwa kupanda miti wakati wa msimu wa mvua ili kulinda vyanzo vya maji na kuzuia ukame.",
        "english": "Protect our environment by planting trees during the rainy season to conserve water sources and prevent drought.",
        "source": "Ministry of Environment 15 Billion Trees Campaign",
        "source_url": "https://environment.go.ke",
        "source_type": "Public Service Announcement (Environmental Conservation)"
    },
    {
        "id": 22,
        "ekegusii": "Abanto bosi abakoro bagosabigwa kororwa oborwaire bwa shukari na obonge bw’amanyinga (pressure) kera ekero gia moka ime y’ospitari erinde banyore okobeyanera ekero bote bwagera.",
        "kiswahili": "Watu wote wazee na wananchi wanaombwa kupimwa sukari na shinikizo la damu (presha) hospitalini ili kupata matibabu ya mapema.",
        "english": "All adults and seniors are encouraged to undergo blood sugar and blood pressure screening at health facilities for early medical management.",
        "source": "Non-Communicable Diseases Alliance & MOH Kenya",
        "source_url": "https://www.health.go.ke",
        "source_type": "Public Service Announcement (Health & Safety)"
    },
    {
        "id": 23,
        "ekegusii": "Abana b’amachora chia sekondari na chiyunisibasi korwa ase amachoka amataka nigo bagosabigwa kwiorikithia ase ekebao kia bursary gia Ward Representative (MCA) obosio bwa tarehe 15.",
        "kiswahili": "Wanafunzi wa shule za sekondari na vyuo vikuu kutoka familia zisizo na uwezo wanaombwa kuomba hazina ya bursary katika ofisi za MCA kabla ya tarehe 15.",
        "english": "Secondary school and university students from needy families are invited to apply for county bursary funds at Ward MCA offices before the 15th.",
        "source": "County Government Bursary Board",
        "source_url": "https://kisii.go.ke",
        "source_type": "Public Service Announcement (Education Bursary)"
    },
    {
        "id": 24,
        "ekegusii": "Abanto bare ne-eching'oso chi'amaiso na okotara koene nigo bagosabigwa gochia ase ospitari ya KTRH tarehe 20 konyora okororwa amaiso na ogosepwa bo-bosa korwa ase abameretiri b'amaiso.",
        "kiswahili": "Watu wenye matatizo ya macho na mtoto wa jicho wanaombwa kufika katika hospitali kuu ya KTRH tarehe 20 kupata uchunguzi na upasuaji wa bure.",
        "english": "People with eye problems and cataracts are invited to attend the main KTRH hospital on the 20th for free eye examination and surgery.",
        "source": "KTRH & SightSavers Free Eye Medical Camp",
        "source_url": "https://ktrh.or.ke",
        "source_type": "Public Service Announcement (Medical Camp)"
    },
    {
        "id": 25,
        "ekegusii": "Abanto bosi basungere chimbwa nigo bagosabigwa kotoera chimbwa chiabo chanjo y’oborwaire bwa rabies ase chi-opisi chia mifugo chia sub-county wiki eyo echete.",
        "kiswahili": "Wafugaji wote wa mbwa wanaombwa kuwapeleka mbwa wao kupata chanjo dhidi ya ugonjwa wa kichaa cha mbwa (rabies) katika ofisi za mifugo wiki ijayo.",
        "english": "All dog owners are asked to take their dogs for rabies vaccination at sub-county livestock offices next week.",
        "source": "Veterinary Services Department Rabies Campaign",
        "source_url": "https://kilimo.go.ke",
        "source_type": "Public Service Announcement (Veterinary Health)"
    },
    {
        "id": 26,
        "ekegusii": "Takatiga riko rira-rogosemba (jiko) ime y’enyomba eyianire ekero mwarete; amaika ya riko agowita abanto ekero bararete obotuko.",
        "kiswahili": "Usiku usiwache jiko la mkaa likiwaka ndani ya nyumba iliyofungwa; moshi wa mkaa unaweza kusababisha vifo kwa kukosa hewa.",
        "english": "Do not leave a burning charcoal stove (jiko) inside a closed room at night; charcoal fumes can cause suffocation and death.",
        "source": "Kenya Red Cross Household Safety Advisory",
        "source_url": "https://redcross.or.ke",
        "source_type": "Public Service Announcement (Household Safety)"
    },
    {
        "id": 27,
        "ekegusii": "Chanjo y'oborwaire bwa rosero (measles na rubella) eyianire koebeyanwa bosa ase abana bosi korwa ameji tano goika emiaka etano ase ebituo bi'obogorwa.",
        "kiswahili": "Chanjo ya surua (measles) inatolewa bure kwa watoto wote kuanzia miezi tisa hadi miaka mitano katika vituo vyote vya afya.",
        "english": "Measles and rubella immunization is provided free of charge for all children aged 9 months to 5 years at all health centers.",
        "source": "WHO & Ministry of Health Immunization Campaign",
        "source_url": "https://www.health.go.ke",
        "source_type": "Public Service Announcement (Child Health)"
    },
    {
        "id": 28,
        "ekegusii": "Abachama bosi b’ekenyoro nigo bagosabigwa gochia ase ebarasa ya Chifu tarehe 10 kwigwa amang'ana y'oborinde na emechando y'amachoka.",
        "kiswahili": "Wakaazi wote wa mtaa wanaombwa kuhudhuria mkutano wa baraza la chifu tarehe 10 kujadili masuala ya usalama na maendeleo ya jamii.",
        "english": "All community residents are requested to attend the Chief's baraza meeting on the 10th to discuss security and local development.",
        "source": "Ministry of Interior Chief's Baraza Notice",
        "source_url": "https://interior.go.ke",
        "source_type": "Public Service Announcement (Civic Affairs)"
    }
]

base_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
psa_data_dir = os.path.join(base_dir, "data", "psa_dataset")
psa_dataset_dir = os.path.join(base_dir, "dataset", "psa_dataset")

# Load existing 12 PSA rows
existing_psa_file = os.path.join(psa_data_dir, "psa_ekegusii_dataset.csv")
df_existing_psa = pd.read_csv(existing_psa_file)
df_new_psa = pd.DataFrame(additional_psa_rows)

df_full_psa = pd.concat([df_existing_psa, df_new_psa], ignore_index=True)
df_full_psa.drop_duplicates(subset=["ekegusii", "english"], inplace=True)
df_full_psa["id"] = range(1, len(df_full_psa) + 1)

print(f"Total PSA dataset rows after expansion: {len(df_full_psa)}")

# Save to psa_dataset directories
df_full_psa.to_csv(os.path.join(psa_data_dir, "psa_ekegusii_dataset.csv"), index=False, encoding="utf-8-sig")
df_full_psa.to_csv(os.path.join(psa_dataset_dir, "psa_ekegusii_dataset.csv"), index=False, encoding="utf-8-sig")

df_full_psa.to_json(os.path.join(psa_data_dir, "psa_ekegusii_dataset.json"), orient="records", indent=2)
df_full_psa.to_json(os.path.join(psa_dataset_dir, "psa_ekegusii_dataset.json"), orient="records", indent=2)

# Update Master Corpus
master_file1 = os.path.join(base_dir, "dataset", "ekegusii_master_dataset.csv")
master_file2 = os.path.join(base_dir, "data", "master_corpus", "ekegusii_master_dataset.csv")

df_master = pd.read_csv(master_file1)
df_combined_master = pd.concat([df_master, df_new_psa], ignore_index=True)
df_combined_master.drop_duplicates(subset=["ekegusii", "english"], inplace=True)
df_combined_master["id"] = range(1, len(df_combined_master) + 1)

df_combined_master.to_csv(master_file1, index=False, encoding="utf-8-sig")
df_combined_master.to_csv(master_file2, index=False, encoding="utf-8-sig")

# Re-run Clean Dataset, Audit, & Splits
dataset_dir = os.path.join(base_dir, "dataset")
splits_dir = os.path.join(dataset_dir, "splits")

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

df_combined_master["audit_flag"] = df_combined_master.apply(check_suspect, axis=1)
df_clean = df_combined_master[df_combined_master["audit_flag"] == "clean"].copy()
df_clean.drop(columns=["audit_flag"], inplace=True)
df_clean.to_csv(os.path.join(dataset_dir, "ekegusii_clean.csv"), index=False)

# Splits
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

print("=== EXPANDED PSA & MASTER DATASET STATS ===")
print(f"Standalone PSA Rows (`psa_dataset/`): {len(df_full_psa)}")
print(f"Master Corpus Total Rows: {len(df_combined_master)}")
print(f"Clean Production Rows: {len(df_clean)}")
print(f"Train split: {len(df_train)} | Val split: {len(df_val)} | Test split: {len(df_test)}")
