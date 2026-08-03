import pandas as pd
import os

def generate_kiswahili_ekegusii_dictionary():
    print("=== Compiling Kiswahili - Ekegusii Parallel Dictionary & Phrasebook ===")
    
    dictionary_entries = [
        # Greetings & Phrases
        {"swahili": "Habari / Jambo", "ekegusii": "Omoiyo / Reroke", "category": "Greetings"},
        {"swahili": "Habari gani?", "ekegusii": "Bwairire naki? / Habari naki?", "category": "Greetings"},
        {"swahili": "Asante sana", "ekegusii": "Mbuya mono", "category": "Phrases"},
        {"swahili": "Karibu", "ekegusii": "Nchwo / Keria", "category": "Phrases"},
        {"swahili": "Kwaheri", "ekegusii": "Tigana buya / Genda buya", "category": "Phrases"},
        {"swahili": "Ndiyo", "ekegusii": "Eee / Yaani", "category": "Phrases"},
        {"swahili": "Hapana", "ekegusii": "Yaya / Yaye", "category": "Phrases"},
        {"swahili": "Tafadhali", "ekegusii": "Kaasero / Nkaasero", "category": "Phrases"},
        {"swahili": "Sijui", "ekegusii": "Tinimanyeti", "category": "Phrases"},
        {"swahili": "Unatoka wapi?", "ekegusii": "Ing’o orwaire?", "category": "Phrases"},
        {"swahili": "Jina lakutwa nani?", "ekegusii": "Rietwa riao ningo?", "category": "Phrases"},
        
        # Family & People
        {"swahili": "Mtu", "ekegusii": "Omonto", "category": "People"},
        {"swahili": "Watu", "ekegusii": "Abanto", "category": "People"},
        {"swahili": "Mwanaume", "ekegusii": "Omosacha", "category": "People"},
        {"swahili": "Mwanamke", "ekegusii": "Omokungu", "category": "People"},
        {"swahili": "Mtoto", "ekegusii": "Omwana", "category": "People"},
        {"swahili": "Watoto", "ekegusii": "Abana", "category": "People"},
        {"swahili": "Baba", "ekegusii": "Tata", "category": "People"},
        {"swahili": "Mama", "ekegusii": "Mama / Baba", "category": "People"},
        {"swahili": "Kaka / Ndugu", "ekegusii": "Momura omwabo / Omonani", "category": "People"},
        {"swahili": "Dada", "ekegusii": "Msubati omwabo / Mosubati", "category": "People"},
        {"swahili": "Rafiki", "ekegusii": "Omosani", "category": "People"},
        {"swahili": "Mwalimu", "ekegusii": "Omwalimu / Omworokia", "category": "People"},
        {"swahili": "Mzee", "ekegusii": "Omugaka", "category": "People"},
        {"swahili": "Kijana", "ekegusii": "Omomura", "category": "People"},
        
        # Numbers
        {"swahili": "Moja", "ekegusii": "Erimo", "category": "Numbers"},
        {"swahili": "Mbili", "ekegusii": "Ebere", "category": "Numbers"},
        {"swahili": "Tatu", "ekegusii": "Etato", "category": "Numbers"},
        {"swahili": "Nne", "ekegusii": "Ene", "category": "Numbers"},
        {"swahili": "Tano", "ekegusii": "Etano", "category": "Numbers"},
        {"swahili": "Sita", "ekegusii": "Etano na erimo (6)", "category": "Numbers"},
        {"swahili": "Saba", "ekegusii": "Etano na ebere (7)", "category": "Numbers"},
        {"swahili": "Nane", "ekegusii": "Enane / Etano na etato", "category": "Numbers"},
        {"swahili": "Tisini / Tisa", "ekegusii": "Kianda", "category": "Numbers"},
        {"swahili": "Kumi", "ekegusii": "Ikomi", "category": "Numbers"},
        {"swahili": "Mia moja", "ekegusii": "Rigana erimo", "category": "Numbers"},
        {"swahili": "Elfu moja", "ekegusii": "Ekerosio erimo", "category": "Numbers"},
        
        # Time & Environment
        {"swahili": "Leo", "ekegusii": "Rero", "category": "Time"},
        {"swahili": "Kesho", "ekegusii": "Ankio", "category": "Time"},
        {"swahili": "Jana", "ekegusii": "Igoro", "category": "Time"},
        {"swahili": "Asubuhi", "ekegusii": "Mambia", "category": "Time"},
        {"swahili": "Mchana", "ekegusii": "Omobaso", "category": "Time"},
        {"swahili": "Jioni", "ekegusii": "Mogoroba", "category": "Time"},
        {"swahili": "Usiku", "ekegusii": "Obotuko", "category": "Time"},
        {"swahili": "Mwaka", "ekegusii": "Omwaka", "category": "Time"},
        {"swahili": "Mwezi", "ekegusii": "Omotienyi", "category": "Time"},
        {"swahili": "Maji", "ekegusii": "Amaache", "category": "Nature"},
        {"swahili": "Moto", "ekegusii": "Omorero", "category": "Nature"},
        {"swahili": "Mti", "ekegusii": "Omote", "category": "Nature"},
        {"swahili": "Ardhi / Nchi", "ekegusii": "Ense", "category": "Nature"},
        {"swahili": "Jua", "ekegusii": "Erioba / Omogaso", "category": "Nature"},
        {"swahili": "Mvua", "ekegusii": "Embura", "category": "Nature"},
        
        # Health & Medical Terms
        {"swahili": "Afya", "ekegusii": "Obochenu / Oboyamu", "category": "Health"},
        {"swahili": "Ugonjwa", "ekegusii": "Oborwaire", "category": "Health"},
        {"swahili": "Hospitali", "ekegusii": "Enyagitari / Sibitari", "category": "Health"},
        {"swahili": "Dawa", "ekegusii": "Omobao / Chindawa", "category": "Health"},
        {"swahili": "Kichwa", "ekegusii": "Omotwe", "category": "Body"},
        {"swahili": "Mkono", "ekegusii": "Omoboko", "category": "Body"},
        {"swahili": "Mguu", "ekegusii": "Ekegoro", "category": "Body"},
        {"swahili": "Jicho", "ekegusii": "Riso", "category": "Body"},
        {"swahili": "Tumbo", "ekegusii": "Ekeuno / Inda", "category": "Body"},
        {"swahili": "Moyo", "ekegusii": "Engoro", "category": "Body"},
        
        # Verbs & Actions
        {"swahili": "Kula", "ekegusii": "Koria", "category": "Verbs"},
        {"swahili": "Kunywa", "ekegusii": "Gonywa", "category": "Verbs"},
        {"swahili": "Kulima", "ekegusii": "Korema", "category": "Verbs"},
        {"swahili": "Kusoma", "ekegusii": "Gosoma", "category": "Verbs"},
        {"swahili": "Kusema / Kuongea", "ekegusii": "Goteba / Gokwana", "category": "Verbs"},
        {"swahili": "Kutembea", "ekegusii": "Gotaara", "category": "Verbs"},
        {"swahili": "Kuja", "ekegusii": "Konchwo", "category": "Verbs"},
        {"swahili": "Kwenda", "ekegusii": "Gogenda", "category": "Verbs"},
        {"swahili": "Kuona", "ekegusii": "Korora", "category": "Verbs"},
        {"swahili": "Kujua", "ekegusii": "Komanya", "category": "Verbs"},
        {"swahili": "Kufanya", "ekegusii": "Gokora", "category": "Verbs"},
        {"swahili": "Kununua", "ekegusii": "Gogoora", "category": "Verbs"},
        {"swahili": "Kuuza", "ekegusii": "Goonia", "category": "Verbs"},
        
        # Food & Agriculture
        {"swahili": "Chakula", "ekegusii": "Endagera", "category": "Food"},
        {"swahili": "Ugali", "ekegusii": "Obokima", "category": "Food"},
        {"swahili": "Mboga", "ekegusii": "Chinyeni", "category": "Food"},
        {"swahili": "Nyama", "ekegusii": "Enyama", "category": "Food"},
        {"swahili": "Maziwa", "ekegusii": "Amabere", "category": "Food"},
        {"swahili": "Ndizi", "ekegusii": "Ematoke", "category": "Food"},
        {"swahili": "Mahindi", "ekegusii": "Ebibori / Amabori", "category": "Food"},
        {"swahili": "Shamba", "ekegusii": "Omogunda", "category": "Agriculture"},
        {"swahili": "Mbolea", "ekegusii": "Esamani", "category": "Agriculture"}
    ]
    
    df = pd.DataFrame(dictionary_entries)
    
    out_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\clean"
    os.makedirs(out_dir, exist_ok=True)
    
    dict_file = os.path.join(out_dir, "Kiswahili_Ekegusii_Dictionary.csv")
    df.to_csv(dict_file, index=False, encoding='utf-8-sig')
    print(f"[OK] Created Kiswahili-Ekegusii Dictionary ({len(df)} entries) -> {dict_file}")

if __name__ == "__main__":
    generate_kiswahili_ekegusii_dictionary()
