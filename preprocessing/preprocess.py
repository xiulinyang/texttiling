from pathlib import Path
import glob
from tqdm import tqdm
import json
import pandas as pd

filename = []
for id, p in tqdm(enumerate(glob.glob('text-segmentation/data/choi/***/**/*.ref', recursive=True))):
    sections = Path(p).read_text().strip().split('==========')
    for section in sections:
        sentences = section.strip().split('\n')
        for i in range(len(sentences)):
            if sentences[i].strip():
                single_sent = {}
                single_sent['genre']='general'
                single_sent['document_id'] = single_sent['genre']+str(id)
                single_sent['sent'] = sentences[i]
                if i==len(sentences)-1:
                    single_sent['boundary'] = 'yes'
                else:
                    single_sent['boundary'] = 'no'
                filename.append(single_sent)


df_file = pd.DataFrame(filename)
df_file.to_csv('general.tsv', sep='\t')




