import numpy as np
import pandas as pd


def check_dataframe(df):
    assert df.iloc[:, 0].max() + 1 == df.iloc[:, 0].nunique()
    assert df.iloc[:, 1].max() + 1 == df.iloc[:, 1].nunique()
    assert (df.iloc[:, 2].diff().iloc[1:] >= 0).all()

    
def load_jodie_data(name):
    with open(name) as fh:
        interactions = []
        features = []
        for i, line in enumerate(fh):
            if i == 0:
                header = line.strip().split(',')
                continue
            uid, iid, ts, state, *feat = line.strip().split(',')
            interactions.append([int(uid), int(iid), float(ts), int(state)])
            features.append([float(v) for v in feat])
    df = pd.DataFrame(interactions, columns=header[:-1]) 
    features = np.asarray(features)
    check_dataframe(df)
    return df, features


def load_recommendation_data(name):
    df = pd.read_csv(name, header=None)
    df.columns = ['user', 'item', 'timestamp']
    df.iloc[:, :2] -= 1
    df.timestamp -= df.timestamp.min()
    check_dataframe(df)
    features = np.zeros((len(df), 1))
    return df, features


def data_split(train_proportion, df, feats):
    df = df.copy()
    num_interactions = len(df)
    train_end_idx = validation_start_idx = int(num_interactions * train_proportion)
    test_start_idx = int(num_interactions * (train_proportion + .1))
    test_end_idx = int(num_interactions * (train_proportion + .2))
    df_train = df.iloc[:train_end_idx]
    df_valid = df.iloc[validation_start_idx:test_start_idx]
    df_test = df.iloc[test_start_idx:test_end_idx]
    feats_train = feats[:train_end_idx]
    feats_valid = feats[validation_start_idx:test_start_idx]
    feats_test = feats[test_start_idx:test_end_idx]
    return df_train, df_valid, df_test, feats_train, feats_valid, feats_test


def recommendation_to_jodie(in_fname, out_fname):
    df, _ = load_recommendation_data(in_fname)
    df.columns = ['user_id', 'item_id', 'timestamp']
    df['state_label'] = 0
    df['separated_list_of_features'] = 0.0
    df.to_csv(out_fname, index=False)
