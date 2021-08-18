import numpy as np
import pandas as pd
import scipy as sc
from scipy import sparse as sp

from prefetch_generator import BackgroundGenerator

from model_utils import *
from data_utils import data_split


def get_dataloaders(df, features, device, ending_time=1., burnin_time=0., alpha=0.99, train_proportion=0.8, coarse=1):
    n_users, n_items = df.iloc[:, :2].max() + 1
    print(f"#Users: {n_users}, #Items: {n_items}, #Interactions: {len(df)}, #Timestamps: {df.timestamp.nunique()}")

    train_df, valid_df, test_df, train_feats, valid_feats, test_feats = data_split(train_proportion, df, features)
    t_max = df.iloc[:, 2].max()
    train_df.iloc[:, 2] = (train_df.iloc[:, 2] // coarse) * coarse / t_max * ending_time + burnin_time
    valid_df.iloc[:, 2] = valid_df.iloc[:, 2] / t_max * ending_time + burnin_time
    test_df.iloc[:, 2] = test_df.iloc[:, 2] / t_max * ending_time + burnin_time
    train_ds = Dataset(train_df, train_feats, n_users, n_items)
    valid_ds = Dataset(valid_df, valid_feats, n_users, n_items, t0=train_ds.unique_ts[-1], adj0=train_ds[-1][2])
    test_ds = Dataset(test_df, test_feats, n_users, n_items, t0=valid_ds.unique_ts[-1], adj0=valid_ds[-1][2])
    print(f"Records Split: {len(train_df), len(valid_df), len(test_df)}")
    print(f"Timestamps Split: {len(train_ds), len(valid_ds), len(test_ds)}")
    train_dl = Dataloader(train_ds, device, alpha)
    valid_dl = Dataloader(valid_ds, device, alpha)
    test_dl = Dataloader(test_ds, device, alpha)
    return train_dl, valid_dl, test_dl


class Dataloader:

    def __init__(self, ds, device, alpha=.9):
        self.ds = ds
        self.device = device
        self.alpha = alpha
    
    def __len__(self):
        return len(self.ds)
    
    def __iter__(self):
        return self.get_iter(0)

    def get_iter(self, start_idx=0):
        return BackgroundGenerator(self._get_iter(start_idx), 1)

    def _get_iter(self, start_idx=0):
        B = None
        for i in range(start_idx, len(self.ds)):
            if B is None:
                t, dt, B, delta_B, users, items, _feats = self.ds.getitem(i, False)
            else:
                B += delta_B
                t, dt, _, delta_B, users, items, _feats = self.ds.getitem(i, True)
            adj = biadjacency_to_laplacian(B) * self.alpha
            i2u_adj, u2i_adj = biadjacency_to_propagation(delta_B)
            adj, i2u_adj, u2i_adj = [sparse_mx_to_torch_sparse_tensor(v).to(self.device) for v in [adj, i2u_adj, u2i_adj]]
            users = torch.from_numpy(users).long().to(self.device)
            items = torch.from_numpy(items).long().to(self.device)
            yield t, dt, adj, i2u_adj, u2i_adj, users, items


class Dataset:
    
    def __init__(self, df, features, n_users, n_items, t0=0., adj0=None):
        self.df = df
        self.features = features
        assert len(self.df) == len(self.features)
        self.n_users = n_users
        self.n_items = n_items
        self.t0 = t0
        self.adj0 = adj0
        self.unique_ts, self.cum_n_records = self.process_timestamps(df.iloc[:, 2])
    
    def __len__(self):
        return len(self.unique_ts)
    
    def __getitem__(self, idx):
        return self.getitem(idx, False)
        
    def getitem(self, idx, only_delta=True):
        t = self.unique_ts[idx]
        dt = t - (self.unique_ts[idx-1] if idx > 0 else self.t0)
        a = self.cum_n_records[idx]
        b = self.cum_n_records[idx+1]
        if only_delta:
            observed_mat = None
        else:
            observed_mat = self.build_ui_mat(self.df.iloc[:a])
            if self.adj0 is not None:
                observed_mat += self.adj0
        delta_mat = self.build_ui_mat(self.df.iloc[a:b])
        users = self.df.iloc[a:b, 0].values
        items = self.df.iloc[a:b, 1].values
        feats = self.features[a:b]
        return t, dt, observed_mat, delta_mat, users, items, feats
    
    def process_timestamps(self, ts):
        unique_ts = np.unique(ts)
        end_idx_ts_dict = {t: i+1 for i, t in enumerate(ts)}
        end_idx = np.array([0] + [end_idx_ts_dict[t] for t in unique_ts])
        return unique_ts, end_idx
    
    def get_observed_interaction_number(self, query_time):
        exclude_query_idx = np.searchsorted(self.unique_ts, query_time, 'left')
        include_query_idx = np.searchsorted(self.unique_ts, query_time, 'right')
        return self.cum_n_records[exclude_query_idx], self.cum_n_records[include_query_idx]
    
    def build_ui_mat(self, df):
        row = df.iloc[:, 0]
        col = df.iloc[:, 1]
        data = np.ones(len(df))
        adj = sp.csc_matrix((data, (row, col)), shape=[self.n_users, self.n_items])
        return adj
        
    def get_observable_graph(self, query_time):
        end_idx, _ = self.get_observed_interaction_number(query_time)
        return self.build_ui_mat(self.df.iloc[:end_idx])
    
    def get_immediate_graph(self, query_time):
        a, b = self.get_observed_interaction_number(query_time)
        return self.build_ui_mat(self.df.iloc[a:b])    
