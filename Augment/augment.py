import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load translation models
bn2en = pipeline("translation", model="csebuetnlp/banglat5_nmt_bn_en")
en2bn = pipeline("translation", model="csebuetnlp/banglat5_nmt_en_bn")

def back_translate(text):
    try:
        # Bengali -> English
        english_text = bn2en(text)[0]['translation_text']
        print(text)
        print('######')
        print(english_text)
        # English -> Bengali
        bangla_paraphrase = en2bn(english_text)[0]['translation_text']
        print('*******')
        print(bangla_paraphrase)
        # Skip if paraphrase is almost identical
        if bangla_paraphrase.strip() == text.strip():
            return None
        return bangla_paraphrase
    except Exception as e:
        print(f"Error in back translation: {e}")
        return None

# Load dataset
file_path = '/home/bxg-server/Mahim/blp25_hatespeech_subtask_1B_train.tsv'
df = pd.read_csv(file_path, sep='\t', keep_default_na=False)

# Prepare IDs
max_existing_id = df['id'].astype(int).max()
next_id = max(max_existing_id + 1, 1000000)

new_rows = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Back-Translating"):
    paraphrased_text = back_translate(row['text'])
    if paraphrased_text:
        new_rows.append({
            'id': next_id,
            'text': paraphrased_text,
            'label': row['label']
        })
        next_id += 1

# Combine original + augmented data
df_augmented = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# Shuffle
df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

# Save output
df_augmented.to_csv(
    '/home/bxg-server/Mahim/output_augmented_buet.tsv',
    sep='\t',
    index=False,
    encoding='utf-8'
)

