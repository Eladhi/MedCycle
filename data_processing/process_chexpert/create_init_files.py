import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import random

# input folder: the output of "process_csv_files.py"
INPUT_FOLDER = '<path to processed csv file folder>'

# output folder
OUTPUT_FOLDER = '<path to final output data>'

PATHOLOGIES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
                  'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                  'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

# for each pathology - first list if exist, second if not
PATHOLOGIES_TO_TEXT = {
    'No Finding': [[''], ['']],
    'Enlarged Cardiomediastinum': [['The mediastinum appears widened.', 'There is enlargement of the cardiac silhouette.', 'Widened mediastinum.'], ['']],
    'Cardiomegaly': [['Enlargement of the cardiac silhouette.', 'Heart size is enlarged.', 'There is mild cardiomegaly.', 'Enlargement of the heart.'], ['Heart size is normal.', 'The cardiac and mediastinal silhouettes are unremarkable.']],
    'Lung Lesion': [['Lung nodules are visible.', 'Pulmonary nodules are visible.'], ['']],
    'Lung Opacity': [['Pulmonary opacification is noted.', 'Lobe opacity is noted.'], ['']],
    'Edema': [['Pulmonary edema is seen.'], ['No pulmonary edema is seen.']],
    'Consolidation': [['There are areas of consolidation.'], ['No focal consolidation.']],
    'Pneumonia': [['Pneumonia is also present.', 'Compatible with pneumonia.'], ['']],
    'Atelectasis': [['There is atelectasis.'], ['']],
    'Pneumothorax': [['There is small pneumothorax.'], ['No pneumothorax.']],
    'Pleural Effusion': [['Pleural effusion is seen.'], ['No pleural effusion.']],
    'Pleural Other': [[''], ['']],
    'Fracture': [['Rib fractures are seen.'], ['']],
    'Support Devices': [['The tube appears to be in correct position.'], ['']],
}

if __name__ == '__main__':

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for s in ['train', 'val', 'test']:
        df = pd.read_csv(os.path.join(INPUT_FOLDER, s + '.csv'))
        df['I Initial Report'] = ''
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            pathology_vec = row[3:17].to_numpy()
            pathology_vec[pathology_vec != pathology_vec] = 0.0  # remove nan
            gen_report = []
            for pi, p in enumerate(PATHOLOGIES):
                if pathology_vec[pi] > 0:
                    t = random.choice(PATHOLOGIES_TO_TEXT[p][0])
                else:
                    t = random.choice(PATHOLOGIES_TO_TEXT[p][1])
                gen_report.append(t)
            df.at[index, 'I Initial Report'] = ' '.join(gen_report).strip().replace('  ', ' ')

        df.to_csv(os.path.join(OUTPUT_FOLDER, s + '.csv'), index=False)
        print('saved to: ' + os.path.join(OUTPUT_FOLDER, s + '.csv'))

    print('Done')
