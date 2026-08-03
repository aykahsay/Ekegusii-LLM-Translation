import shutil
import os

def mirror_notebook():
    src_nb = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\notebooks\A100_3_Architectures_Training.ipynb"
    dest_nb = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\translation_model_last.ipynb"
    dest_nb2 = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\notebooks\translation_model_last.ipynb"
    
    if os.path.exists(src_nb):
        shutil.copy2(src_nb, dest_nb)
        shutil.copy2(src_nb, dest_nb2)
        print(f"[OK] Mirrored master notebook to {dest_nb} and {dest_nb2}")

if __name__ == "__main__":
    mirror_notebook()
