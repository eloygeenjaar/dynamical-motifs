import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold


def get_icafbirn(seed, fold=0):
    df = pd.read_csv('/data/users1/egeenjaar/local-global/data/ica_fbirn/info_df.csv', index_col=0)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    y = df['sz'].values
    splits = skf.split(df.index.values, y)
    splits = [split for split in splits]
    trainval_index, test_index = splits[fold]
    train_index, valid_index = train_test_split(
        trainval_index, train_size=0.9, random_state=seed,
        stratify=y[trainval_index])
    train_df = df.iloc[train_index].copy()
    valid_df = df.iloc[valid_index].copy()
    test_df = df.iloc[test_index].copy()
    return train_df, valid_df, test_df

def get_ukbb(seed, fold=0):
    data_p = Path('/data/qneuromark/Results/ICA/UKBioBank')
    subjects = list(data_p.iterdir())
    subjects = [(subject / 'UKB_sub01_timecourses_ica_s1_.nii') for subject in subjects]
    subjects = [subject for subject in subjects if subject.is_file()]
    trainvalid_subjects, test_subjects = train_test_split(subjects, train_size=0.9, random_state=seed, shuffle=True)
    train_subjects, valid_subjects = train_test_split(trainvalid_subjects, train_size=0.9, random_state=seed, shuffle=True)
    return train_subjects, valid_subjects, test_subjects
    