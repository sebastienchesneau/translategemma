from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
import torch
import os
import time

# Chargement du modèle et du processeur
model_name = "facebook/seamless-m4t-v2-large"
processor = AutoProcessor.from_pretrained(model_name)
model = SeamlessM4Tv2ForTextToText.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
folder = "./samples"

# Langues (ISO 639-3)
src_lang = "fra"
target_langs = ["fra", "eng", "deu", "cmn", "spa"]

for filename in os.listdir(folder):
    print(f"Traduction ssssss")
    if filename.lower().endswith(".txt"):
        file_path = os.path.join(folder, filename)
        print(f"Traduction de : {filename}...")

        with open(file_path, "r", encoding="utf-8") as f:
          input_text = f.read().strip()

        # start = time.time()
        # Préparation de l'entrée
        inputs = processor(text=input_text, src_lang=src_lang, return_tensors="pt").to(device)

        # Génération (traduction)
        for tgt_lang in target_langs:
            start = time.perf_counter()
            with torch.no_grad():
                generated_tokens = model.generate(**inputs, tgt_lang=tgt_lang)
                # Décodage du résultat
                output_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                print(f"Traduction: {output_text}")
                end = time.perf_counter()
                print(f"Temps d'exécution : {end - start:.6f} secondes")

        # print(f"✅ Traduit en {str(time.time() - start)} s")
