import os
import pandas as pd

print("=== Creating Dedicated PSA Dataset Folder & Files ===")

psa_rows = [
    {
        "id": 1,
        "ekegusii": "Tiga tosabie amaboko n'amasabi goetera amache amachenu chinsaku chionsi obosio bw'okoria, obosio bw'okomora omwana na ekero waeta aaria choo erinde totebia oborwaire bwa korwa-n'okoyoria (cholera).",
        "kiswahili": "Tafadhali osha mikono yako kwa sabuni na maji safi kila wakati kabla ya kula, kabla ya kumlisha mtoto, na baada ya kutumia choo ili kuzuia ugonjwa wa kipindupindu.",
        "english": "Please wash your hands with soap and clean water at all times before eating, before feeding a child, and after using the toilet to prevent cholera.",
        "source": "Ministry of Health & Kisii County Health Public Service Announcements",
        "source_url": "https://www.kisii.go.ke/departments/health-services",
        "source_type": "Public Service Announcement (Health & Safety)"
    },
    {
        "id": 2,
        "ekegusii": "Ababari bosi nagosabigwa bare batwale abana babo abake inse y'emiaka etano ase ebituo bi'obogorwa obosio bwa chanjo y'oborwaire bwa polio erinde barinde abana babo korwa ase oborwaire obwo obebe.",
        "kiswahili": "Wazazi wote wanaombwa kuwapeleka watoto wao walio chini ya miaka mitano kwenye vituo vya afya kwa ajili ya chanjo ya polio ili kuwalinda na ugonjwa huo hatari.",
        "english": "All parents are urged to take their children under five years old to health centers for polio vaccination to protect them from the dangerous disease.",
        "source": "Ministry of Health & Kisii County Health Public Service Announcements",
        "source_url": "https://www.kisii.go.ke/departments/health-services",
        "source_type": "Public Service Announcement (Health & Safety)"
    },
    {
        "id": 3,
        "ekegusii": "Abasubati bosi abaito bare nechenda nigo bagosabigwa gochia ase ebituo bi'obogorwa konyora ekero gia klinik kera omomura erinde barindwe na barinde abana babo goikera ekero gia koibora.",
        "kiswahili": "Wanawake wote wajawazito wanaombwa kuhudhuria kliniki ya afya kila mwezi ili kupata uchunguzi na kuhakikisha usalama wao na wa watoto wao hadi wakati wa kujifungua.",
        "english": "All pregnant women are urged to attend health clinics every month for medical checkups to ensure their safety and the safety of their unborn babies until delivery.",
        "source": "Ministry of Health & Kisii County Health Public Service Announcements",
        "source_url": "https://www.kisii.go.ke/departments/health-services",
        "source_type": "Public Service Announcement (Health & Safety)"
    },
    {
        "id": 4,
        "ekegusii": "Komonyora karia chinyusi chi'ebiyobi korwa ase omosunte goika torare ime y'ebitanda bi'etabu y'ebiyobi ekero tore korenyora obotuko ase enyomba.",
        "kiswahili": "Ili kujikinga na mbu wanaoeneza homa ya malaria, hakikisha unalala ndani ya neti iliyotiwa dawa kila usiku.",
        "english": "To protect yourself from mosquitoes spreading malaria, make sure you sleep under a treated mosquito net every night.",
        "source": "Ministry of Health & Kisii County Health Public Service Announcements",
        "source_url": "https://www.kisii.go.ke/departments/health-services",
        "source_type": "Public Service Announcement (Health & Safety)"
    },
    {
        "id": 5,
        "ekegusii": "Abaramura na abaiseke bosi baikize emiaka ikumi na enaate (18) nigo bagosabigwa kwiorikithia ase ekebao kia IEBC erinde banyore rigoti ri'ogotora ase ekero kia chichura.",
        "kiswahili": "Vijana wote waliotimiza miaka kumi na nane (18) wanaombwa kujisajili kama wapiga kura kwenye vituo vya IEBC ili kupata haki ya kupiga kura katika uchaguzi.",
        "english": "All youth who have reached eighteen (18) years of age are urged to register as voters at IEBC centers to get the right to vote in the election.",
        "source": "IEBC Civic Education Public Announcement",
        "source_url": "https://www.iebc.or.ke",
        "source_type": "Public Service Announcement (Civic Duty)"
    },
    {
        "id": 6,
        "ekegusii": "Abaserikali na abagendi b'ebibosibosi (boda boda) nigo bagochigwa korwata ekepera (kofia ya chuma) ekero bare gotwara gose korigora erinde berinde korwa ase ebireng'u bi'enchera.",
        "kiswahili": "Madereva na abiria wa pikipiki (boda boda) wanatakiwa kuvaa kofia ngumu (helimeti) kila wakati wanaposafiri ili kujilinda dhidi ya ajali za barabarani.",
        "english": "Motorcycle riders and passengers (boda boda) are required to wear helmets at all times when traveling to protect themselves against road accidents.",
        "source": "NTSA Road Safety Public Service Announcement",
        "source_url": "https://ntsa.go.ke",
        "source_type": "Public Service Announcement (Road Safety)"
    },
    {
        "id": 7,
        "ekegusii": "Omwana omoke nigo akweneretie Konymka amabeera y'omonyene (amabeera y'omoseke) boka ase emiefe etano na eyemo y'otang'ani otatiga koria endagera eyende yonsi erinde anyore obogoro bw'omobere.",
        "kiswahili": "Mtoto mchanga anapaswa kunyonyeshwa maziwa ya mama pekee kwa miezi sita ya kwanza bila kupewa chakula kingine ili kupata afya bora na kinga mwilini.",
        "english": "Infants should be exclusively breastfed with mother's milk for the first six months without giving any other food to ensure optimal health and immunity.",
        "source": "UNICEF & Ministry of Health Nutrition Advisory",
        "source_url": "https://www.unicef.org/kenya",
        "source_type": "Public Service Announcement (Nutrition)"
    },
    {
        "id": 8,
        "ekegusii": "Ekero embura enene egwa, dua egechano nigo togosabigwa koba ase oborinde obuya, totereng’ana chiremo na chinyancha chi'amache aya-are obonge nengencho y'amache omororo.",
        "kiswahili": "Wakati wa mvua kubwa na mafuriko, wananchi wanaombwa kukaa mahali salama na kuepuka kuvuka mito au maeneo yenye maji mengi ili kuzuia maafa.",
        "english": "During heavy rains and floods, citizens are urged to stay in safe places and avoid crossing rivers or areas with high water levels to prevent disaster.",
        "source": "National Disaster Management Authority Public Notice",
        "source_url": "https://ndma.go.ke",
        "source_type": "Public Service Announcement (Disaster Preparedness)"
    },
    {
        "id": 9,
        "ekegusii": "Omonto oyomo oyore nogokorora goetania chibiki ibere nigo akweneretie gochia ase ekebao kia ospitari konyora okororwa oborwaire bwa TB boigo ogoterwa embere tare koambwerania ase amachoka.",
        "kiswahili": "Mtu yeyote anayekohoa kwa zaidi ya wiki mbili anapaswa kwenda hospitalini kupimwa ugonjwa wa kifua kikuu (TB) na kuanza matibabu bila malipo.",
        "english": "Anyone coughing for more than two weeks should go to the hospital to get tested for tuberculosis (TB) and start free treatment.",
        "source": "Ministry of Health TB Prevention Campaign",
        "source_url": "https://www.health.go.ke",
        "source_type": "Public Service Announcement (Disease Prevention)"
    },
    {
        "id": 10,
        "ekegusii": "Oborwaire bw'omoyo na obochoro bw'omoraigwa nobong'aini boene oborio abanto babwate; inee orakora obosani, abamoni na abasaani babo tebia abanto basomerie obogoro bw'omoyo erinde banyore okoreterwa.",
        "kiswahili": "Afya ya akili ni muhimu kwa kila mtu; ikiwa unahisi msongo wa mawazo au huzuni, tafadhali zungumza na mshauri wa afya au mtu unayemwamini ili kupata msaada.",
        "english": "Mental health is essential for everyone; if you feel stressed or depressed, please speak to a counselor or trusted person to get support.",
        "source": "Ministry of Health Mental Health Advisory",
        "source_url": "https://www.health.go.ke",
        "source_type": "Public Service Announcement (Mental Health)"
    },
    {
        "id": 11,
        "ekegusii": "Abakungu na abasacha bosi nigo bagosabigwa gochia ase chinospitari chia kaunti gokorwa okororwa koochi kwerinda korwa ase oborwaire bwa kansa obosio bwokoria ekero bote bwagera.",
        "kiswahili": "Wanawake na wanaume wote wanaombwa kwenda katika hospitali za kaunti kufanyiwa uchunguzi wa mapema wa saratani (kansa) ili kupata matibabu kwa wakati.",
        "english": "All women and men are urged to visit county hospitals for early cancer screening to receive timely treatment.",
        "source": "National Cancer Control Program Public Notice",
        "source_url": "https://www.health.go.ke",
        "source_type": "Public Service Announcement (Cancer Screening)"
    },
    {
        "id": 12,
        "ekegusii": "Kera amachoka na kera enyomba nigo ekweneretie koba nechoo eyianire obochenu obuya erinde torinde chinsemo chiito korwa ase ebiseere na amagano y'echinse.",
        "kiswahili": "Kila kaya inapaswa kuwa na choo safi na chenye kifuniko ili kulinda mazingira yetu dhidi ya inzi na magonjwa ya tumbo.",
        "english": "Every household should have a clean and covered toilet to protect our environment from flies and diarrheal diseases.",
        "source": "Public Health & Sanitation Department Advisory",
        "source_url": "https://www.kisii.go.ke/departments/health-services",
        "source_type": "Public Service Announcement (Sanitation)"
    }
]

df_psa = pd.DataFrame(psa_rows)

base_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
psa_data_dir = os.path.join(base_dir, "data", "psa_dataset")
psa_dataset_dir = os.path.join(base_dir, "dataset", "psa_dataset")

os.makedirs(psa_data_dir, exist_ok=True)
os.makedirs(psa_dataset_dir, exist_ok=True)

file1 = os.path.join(psa_data_dir, "psa_ekegusii_dataset.csv")
file2 = os.path.join(psa_dataset_dir, "psa_ekegusii_dataset.csv")

df_psa.to_csv(file1, index=False, encoding="utf-8-sig")
df_psa.to_csv(file2, index=False, encoding="utf-8-sig")

# Create JSON format as well for easy access
df_psa.to_json(os.path.join(psa_data_dir, "psa_ekegusii_dataset.json"), orient="records", indent=2)
df_psa.to_json(os.path.join(psa_dataset_dir, "psa_ekegusii_dataset.json"), orient="records", indent=2)

print(f"Exported PSA dataset ({len(df_psa)} rows) to:")
print(f" 1. {file1}")
print(f" 2. {file2}")
print(f" 3. JSON representations created.")
