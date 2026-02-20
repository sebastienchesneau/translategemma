from transformers import pipeline
import torch
import time

pipe = pipeline(
    "image-text-to-text",
    model="google/translategemma-4b-it",
    device="cuda",
    dtype=torch.bfloat16
)


def translate_text_pipeline(
    text: str,
    source_lang: str,
    target_lang: str,
    max_new_tokens: int = 512,
) -> str:
    # ---- Text Translation ----
    start = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang,
                    "target_lang_code": target_lang,
                    "text": text,
                },
            ],
        }
    ]

    output = pipe(text=messages, max_new_tokens=max_new_tokens)
    print(output[0]["generated_text"][-1]["content"])
    end = time.perf_counter()
    print(f"Temps d'exécution : {end - start:.6f} secondes")


examples = [
    ("Bonjour, comment allez-vous ?", "fr", "en"),
    ("La science est fascinante.", "fr", "de"),
    ("Guten Morgen, wie geht es Ihnen?", "de", "fr"),
    ("今日はとても暑いです。", "ja", "fr"),
    ("je suis en train de faire du rangement dans le grenier de ma grand mère", "fr", "de"),
    ("Ich räume den Dachboden meiner Großmutter auf.", "de", "fr"),
    ('Der Morgen beginnt mit einem sanften Licht am Himmel.', "de", "fr"),
    ("Die Vögel singen leise und die Stadt erwacht langsam.", "de", "fr"),
    ("Ich trinke meinen Kaffee und denke über den Tag nach.", "de", "fr"),
    ("Neue Aufgaben warten und bringen spannende Herausforderungen.", "de", "fr"),
    ("Am Abend hoffe ich, zufrieden und glücklich einschlafen zu können.", "de", "fr"),
]

for text, src, tgt in examples:
    translate_text_pipeline(text, src, tgt, 200)






# # ---- Text Extraction and Translation ----
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "source_lang_code": "cs",
#                 "target_lang_code": "de-DE",
#                 "url": "https://c7.alamy.com/comp/2YAX36N/traffic-signs-in-czech-republic-pedestrian-zone-2YAX36N.jpg",
#             },
#         ],
#     }
# ]

# output = pipe(text=messages, max_new_tokens=200)
# print(output[0]["generated_text"][-1]["content"])
