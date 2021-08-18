import torch
from torch import nn
from torch.nn import functional as F

from acgnn import ACGNN


class CoPE(nn.Module):
    
    def __init__(self, n_users, n_items, hidden_size, n_neg_samples=16):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.n_neg_samples = n_neg_samples
        self.user_states = nn.Parameter(torch.rand(n_users, hidden_size))
        self.item_states = nn.Parameter(torch.rand(n_items, hidden_size))
        trunc_normal_(self.user_states.data, std=0.01)
        trunc_normal_(self.item_states.data, std=0.01)
        self.propagate_unit = PropagateUnit(n_users, n_items)
        self.update_unit = UpdateUnit(hidden_size)
        self.u_pred_mapping = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.i_pred_mapping = nn.Linear(2 * hidden_size, 2 * hidden_size)
        nn.init.eye_(self.u_pred_mapping.weight.data)
        nn.init.eye_(self.i_pred_mapping.weight.data)
        nn.init.zeros_(self.u_pred_mapping.bias.data)
        nn.init.zeros_(self.i_pred_mapping.bias.data)
    
    def get_init_states(self):
        return self.user_states, self.item_states
    
    def forward(self, adj, dt, last_xu, last_xi):
        return self.propagate(adj, dt, last_xu, last_xi)

    def propagate(self, adj, dt, last_xu, last_xi):
        yu, yi = self.propagate_unit(adj, dt, last_xu, last_xi, self.user_states, self.item_states)
        return yu, yi
    
    def propagate_update(self, adj, dt, last_xu, last_xi, i2u_adj, u2i_adj):
        yu, yi = self.propagate(adj, dt, last_xu, last_xi)
        zu, zi, _ = self.update_unit(yu, yi, i2u_adj, u2i_adj)
        return yu, yi, zu, zi

    def compute_matched_scores(self, hu, hi):
        hu = self.u_pred_mapping(hu)
        hi = self.i_pred_mapping(hi)
        eps = 1e-8
        return (hu * hi).sum(1)

    def compute_pairwise_scores(self, hu, hi):
        hu = self.u_pred_mapping(hu)
        hi = self.i_pred_mapping(hi)
        eps = 1e-8
        hu = hu.unsqueeze(1)  # [m, 1, d]
        hi = hi.unsqueeze(0)  # [1, n, d]
        return (hu * hi).sum(2)

    def compute_loss(self, yu, yi, users, items):
        mn = None
        n = len(users)
        yu = torch.cat([yu, self.user_states], 1)
        yi = torch.cat([yi, self.item_states], 1)
        # positive
        pos_u = F.embedding(users, yu, max_norm=mn)
        pos_i = F.embedding(items, yi, max_norm=mn)
        pos_scores = self.compute_matched_scores(pos_u, pos_i)
        # negative
        neg_u_ids = torch.randint(0, self.n_users, size=[self.n_neg_samples//2], device=users.device)
        neg_i_ids = torch.randint(0, self.n_items, size=[self.n_neg_samples//2], device=items.device)
        neg_u = F.embedding(neg_u_ids, yu, max_norm=mn)
        neg_i = F.embedding(neg_i_ids, yi, max_norm=mn)
        u_neg_scores = self.compute_pairwise_scores(pos_u, neg_i)
        i_neg_scores = self.compute_pairwise_scores(neg_u, pos_i)
        neg_scores = torch.cat([u_neg_scores, i_neg_scores.T], 1)
        scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], 1)
        logps = F.log_softmax(scores, 1)
        loss = -logps[:, 0].mean()
        return loss
    
    def propagate_update_loss(self, adj, dt, last_xu, last_xi, i2u_adj, u2i_adj, users, items):
        # propagate
        yu, yi = self.propagate(adj, dt, last_xu, last_xi)
        # compute loss
        loss = self.compute_loss(yu, yi, users, items)
        # update and return
        zu, zi, delta_norm = self.update_unit(yu, yi, i2u_adj, u2i_adj)
        return loss, delta_norm, zu, zi, yu, yi


class PropagateUnit(nn.Module):
    
    def __init__(self, n_users, n_items):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.gnn = ACGNN(10, 2, self.n_users + self.n_items, True)
    
    def forward(self, adj, dt, xu, xi, static_u, static_i):
        last_state = torch.cat([xu, xi], 0)
        init_state = torch.cat([static_u, static_i], 0)
        norm = torch.norm(last_state, dim=1).max()
        last_state = last_state / norm
        init_state = init_state / norm
        z = self.gnn(adj, init_state, last_state, dt)
        yu, yi = torch.split(z, [self.n_users, self.n_items], 0)
        return yu, yi


class UpdateUnit(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()
        self.uu_mapping = nn.Linear(hidden_size, hidden_size)
        self.ii_mapping = nn.Linear(hidden_size, hidden_size)
        self.ui_mapping = nn.Linear(hidden_size, hidden_size, bias=False)
        self.iu_mapping = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, user_embs, item_embs, i2u_prop_mat, u2i_prop_mat):
        # user_embs: [m, d]
        # item_embs: [n, d]
        # u2i_prop_mat: [n, m]
        # i2u_prop_mat: [m, n]
        # act_fn = torch.tanh
        act_fn = F.relu
        delta_u = act_fn(self.uu_mapping(user_embs) + i2u_prop_mat @ self.iu_mapping(item_embs))
        delta_i = act_fn(self.ii_mapping(item_embs) + u2i_prop_mat @ self.ui_mapping(user_embs))
        u_mask = (torch.sparse.sum(i2u_prop_mat, 1).to_dense() > 0).float()
        i_mask = (torch.sparse.sum(u2i_prop_mat, 1).to_dense() > 0).float()
        delta_u = delta_u * u_mask.unsqueeze(1)
        delta_i = delta_i * i_mask.unsqueeze(1)
        new_user_embs = user_embs + delta_u
        new_item_embs = item_embs + delta_i
        delta_norm = (delta_u ** 2).sum() / u_mask.sum() + (delta_i ** 2).sum() / i_mask.sum()
        return new_user_embs, new_item_embs, delta_norm


def trunc_normal_(x, mean=0., std=1.):
    # From Fast.ai
    return x.normal_().fmod_(2).mul_(std).add_(mean)
