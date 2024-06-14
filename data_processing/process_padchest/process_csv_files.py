import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

# output folder
ONLY_FRONTAL = False
OUTPUT_FOLDER = '<path to output folder>'

# after running process_mimic_csv.py to generate the following .csv files
MIMICCXR_TRAIN_CSV = '<mimic path>/train.csv'
MIMICCXR_VAL_CSV = '<mimic path>/val.csv'
MIMICCXR_TEST_CSV = '<mimic path>/test.csv'

# raw .csv files of padchest
PADCHEST_TRAIN_CSV = '<padchest path>/train.csv'
PADCHEST_DIR = ''

# chexpert template; for padchest the values of these columns are nan
IMG_PATHOLOGIES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
                  'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                  'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

REPORT_PATHOLOGIES = IMG_PATHOLOGIES


def mimic_df_only(dataframe, N_reports, out_dir, out_file):
    print('Generating mimic-only data')
    df_list = []
    for ri in tqdm(range(N_reports)):
        row_report = dataframe.iloc[ri]
        e = {'Pair': 1}
        e['Image Path'] = row_report['Path']
        for p in REPORT_PATHOLOGIES:
            e['I ' + p] = row_report[p]
        e['N Pathologies Image'] = len(REPORT_PATHOLOGIES)
        e['Report'] = row_report['Report']
        for p in REPORT_PATHOLOGIES:
            e['R ' + p] = row_report[p]
        e['N Pathologies Report'] = len(REPORT_PATHOLOGIES)
        df_list.append(e)

    df = pd.DataFrame(df_list)
    for d in out_dir:
        out_path = os.path.join(OUTPUT_FOLDER, d)
        os.makedirs(out_path, exist_ok=True)
        df.to_csv(os.path.join(out_path, out_file))
    df_list = [df]
    del df_list


if __name__ == '__main__':

    print('Using .csv mimic-cxr files')
    df_report_train = pd.read_csv(MIMICCXR_TRAIN_CSV)

    df_img_train = pd.read_csv(PADCHEST_TRAIN_CSV)

    df_img_train = df_img_train.drop(columns=['AP/PA'])
    if ONLY_FRONTAL:
        df_img_train.drop(df_img_train[df_img_train['Frontal/Lateral'] == 'LATERAL'].index, inplace=True)  # remove lateral images

    ## create the unsupervised training .csv file
    N_images = len(df_img_train)
    N_reports = len(df_report_train)
    print("# images: %d", N_images)
    print("# reports: %d", N_reports)

    img_idx = np.random.permutation(N_images)
    report_idx = np.random.permutation(N_reports)

    # there are ~x2.3 reports than images in these datasets
    img_idx = np.concatenate((img_idx, img_idx, img_idx[:N_reports - 2 * N_images]))

    df_list = []
    for ii, ri in tqdm(zip(img_idx, report_idx), total=img_idx.shape[0]):
        row_image = df_img_train.iloc[ii]
        row_report = df_report_train.iloc[ri]
        e = {'Pair': 0}
        e['Image Path'] = os.path.join(PADCHEST_DIR, row_image['Path'])
        for p in IMG_PATHOLOGIES:
            e['I ' + p] = row_image[p]
        e['N Pathologies Image'] = len(IMG_PATHOLOGIES)
        e['Report'] = row_report['Report']
        for p in REPORT_PATHOLOGIES:
            e['R ' + p] = row_report[p]
        e['N Pathologies Report'] = len(REPORT_PATHOLOGIES)
        e['I Labels'] = row_image['Labels']
        e['I LabelsLocalizationsBySentence'] = row_image['LabelsLocalizationsBySentence']
        e['I Report'] = row_image['Report']
        df_list.append(e)

    df = pd.DataFrame(df_list)
    out_path = os.path.join(OUTPUT_FOLDER, 'unpaired')

    os.makedirs(out_path, exist_ok=True)
    df.to_csv(os.path.join(out_path, 'train.csv'))
    print('Deleting DF')
    df_list = [df]
    del df_list
    print('Done Deleting DF')

    ## create a supervised (mimic-only) .csv file
    mimic_df_only(df_report_train, len(df_report_train), ['paired'], 'train.csv')

    ## create the val (mimic-only) .csv file
    df_report_val = pd.read_csv(MIMICCXR_VAL_CSV)
    mimic_df_only(df_report_val, len(df_report_val), ['paired', 'unpaired'], 'val.csv')

    ## create the test (mimic-only) .csv file
    df_report_test = pd.read_csv(MIMICCXR_TEST_CSV)
    mimic_df_only(df_report_test, len(df_report_test), ['paired', 'unpaired'], 'test.csv')

    print('Done')
