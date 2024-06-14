import pandas as pd
import os
from tqdm import tqdm

PADCHEST_CSV_PATH = '<path to padchest folder>/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz'
OUTPUT_CSV_PATH = '<path to output processed padchest>/train.csv'
IMG_DIR = '<path to padchest image folder>'

# Irrelevant for padchest, it's just for a unified structure with chexpert
PATHOLOGIES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
                  'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                  'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

source_df = pd.read_csv(PADCHEST_CSV_PATH)
output_list = []

for index, row in tqdm(source_df.iterrows(), total=source_df.shape[0]):
    path = os.path.join('train', str(row['ImageDir']), row['ImageID'])
    if not os.path.isfile(os.path.join(IMG_DIR, path)):
        print('skipping + ' + path)
        continue
    fl = row['ViewPosition_DICOM']
    ap = row['Projection']
    labels = row['Labels']
    report = row['Report']
    labels_loc = row['LabelsLocalizationsBySentence']
    dict = {'Path': path, 'Frontal/Lateral': fl, 'AP/PA': ap,
            'Labels': labels, 'LabelsLocalizationsBySentence': labels_loc,
            'Report': report}
    for p in PATHOLOGIES:
        dict[p] = 'nan'
    output_list.append(dict)

output_df = pd.DataFrame(output_list)
output_df.to_csv(OUTPUT_CSV_PATH, index=False)

