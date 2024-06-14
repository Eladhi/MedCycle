import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json

# input folder: the output of "process_csv_files.py"
INPUT_FOLDER = '<path to processed csv file folder>'

# output folder
OUTPUT_FOLDER = '<path to final output data>'

PATHOLOGIES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
                  'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                  'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

def gen_pseudo_report(labeloc, labels):
    # deal with nans, string etc
    if isinstance(labeloc, float):
        labeloc = "['na']"
    if isinstance(labels, float):
        labels = "['na']"
    labeloc = labeloc.replace('nan,', '')
    labeloc = json.loads(labeloc.replace('\'', '"'))
    labels = json.loads(labels.replace('\'', '"'))

    r = []

    # fix data
    if not isinstance(labeloc[0], list) and len(labeloc) == 1:
        labeloc = [labeloc]
    elif not isinstance(labeloc[0], list):
        sublists = []
        sublist = []
        for i in range(len(labeloc)):
            if len(sublist) == 0:
                sublist.append(labeloc[i])
                continue
            if labeloc[i].startswith('loc '):
                sublist.append(labeloc[i])
                continue
            sublists.append(sublist)
            sublist = []
            sublist.append(labeloc[i])
        if len(sublist) > 0:
            sublists.append(sublist)
        labeloc = sublists

    # parse
    for s in labeloc:
        if len(s) == 1 and (s[0] == 'unchanged' or s[0] == 'normal' or s[0] == 'exclude'):
            continue
        si = []
        for w in s:
            if w.startswith('loc ') and w[4:] in s[0]:
                continue
            si.append(w.replace('loc ', ''))
        si = si[::-1]
        si.append('.')
        r.extend(si)

    # add common phrases
    if 'pneumothorax' not in labels:
        r.extend(['no', 'pneumothorax', '.'])
    if 'pleural effusion' not in labels:
        r.extend(['no', 'pleural', 'effusion', '.'])
    if 'cardiomegaly' not in labels:
        r.extend(['heart', 'size', 'is', 'normal', '.'])

    return ' '.join(r).strip().replace('  ', ' ')


if __name__ == '__main__':

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for s in ['train', 'val', 'test']:
        df = pd.read_csv(os.path.join(INPUT_FOLDER, s + '.csv'))
        df['I Initial Report'] = ''
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            pathology_vec = row[3:17].to_numpy()
            pathology_vec[pathology_vec != pathology_vec] = 0.0  # remove nan
            if s == 'train':
                pseudo_report = gen_pseudo_report(row['I LabelsLocalizationsBySentence'], row['I Labels'])
                df.at[index, 'I Initial Report'] = pseudo_report

        if s == 'train':
            df = df.drop(columns=['I Labels', 'I LabelsLocalizationsBySentence', 'I Report'])
        df.to_csv(os.path.join(OUTPUT_FOLDER, s + '.csv'), index=False)
        print('saved to: ' + os.path.join(OUTPUT_FOLDER, s + '.csv'))

    print('Done')
