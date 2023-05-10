import pandas as pd
from tqdm import tqdm
import os

def context(DATASET):
    TMP = DATASET.copy()

    lst = {}
    id = DATASET.iloc[0]['id']
    for i in tqdm(range(1, len(DATASET))):
        speaker = DATASET.iloc[i]['speaker']
        if speaker not in lst:
            lst[speaker] = []

        if (DATASET.iloc[i]['speaker'] == DATASET.iloc[i-1]['speaker']) and (DATASET.iloc[i]['id'] == DATASET.iloc[i-1]['id']):
            j = i-1
            lst[speaker] = []
            while j >= 0:
                if DATASET.iloc[j]['speaker'] == DATASET.iloc[i]['speaker'] and len(lst[speaker]) < 3:
                    source_value = DATASET.iloc[j]['source']
                    if isinstance(source_value, (int, float)):
                        source_value = str(source_value)
                    lst[speaker].append(source_value)
                    j -= 1
                else:
                    break

        s = ""

        if len(lst[speaker]) > 0:
            s = "<context> "
            s = s + " ".join(lst[speaker][::-1])
            s = s + " <end> "
            s = s + str(DATASET.iloc[i]['source'])
            lst[speaker] = []
        else:
            s = str(DATASET.iloc[i]['source'])
        TMP.iloc[i, TMP.columns.get_loc('source')] = s
    return TMP

def process_dataset(input_filename, output_prefix, target_lang):
    data = pd.read_csv(input_filename, sep="\t", names=['id', 'speaker', 'source', 'target'])

    print("Please wait.... This may take a while....")
    data_with_context = context(data.copy())

    if target_lang == 'en':
        with open(output_prefix + '.hi', 'w', encoding='utf-8') as f:
            f.write(data_with_context['source'].astype(str).str.cat(sep='\n'))

        with open(output_prefix + '.en', 'w', encoding='utf-8') as f:
            f.write(data_with_context['target'].astype(str).str.cat(sep='\n'))
    else:
        print("Invalid target lang")

print("Processing Hinglish-English subset")
if not os.path.isdir('context'):
    os.mkdir('context')
process_dataset('test.csv', 'context/test', 'en')
process_dataset('valid.csv', 'context/valid', 'en')
process_dataset('train.csv', 'context/train', 'en')
