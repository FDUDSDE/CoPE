import time
from copy import deepcopy

import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F

from cope import CoPE
from model_utils import *
from eval_utils import *


def train_one_epoch(model, optimizer, train_dl, delta_coef=1e-5, tbptt_len=20,
                    valid_dl=None, test_dl=None, fast_eval=True, adaptation=False, adaptation_lr=1e-4):
    last_xu, last_xi = model.get_init_states()
    loss_pp = 0.
    loss_norm = 0.
    optimizer.zero_grad()
    model.train()
    counter = 0
    pbar = tqdm.tqdm(train_dl)
    cum_loss = 0.
    for i, batch in enumerate(pbar):
        t, dt, adj, i2u_adj, u2i_adj, users, items = batch
        step_loss, delta_norm, last_xu, last_xi, *_ = model.propagate_update_loss(adj, dt, last_xu, last_xi, i2u_adj, u2i_adj, users, items)
        loss_pp += step_loss
        loss_norm += delta_norm
        counter += 1
        if (counter % tbptt_len) == 0 or i == (len(train_dl) - 1):
            total_loss = (loss_pp + loss_norm * delta_coef) / counter
            total_loss.backward()
            optimizer.step()
            cum_loss += total_loss.item()
            pbar.set_description(f"Loss={cum_loss/i:.4f}")
            last_xu = last_xu.detach()
            last_xi = last_xi.detach()
            optimizer.zero_grad()
            loss_pp = 0.
            loss_norm = 0.
            counter = 0
    pbar.close()
    if fast_eval:
        rollout_evaluate_fast(model, valid_dl, test_dl, last_xu.detach(), last_xi.detach())
    else:
        rollout_evaluate(model, train_dl, valid_dl, test_dl)


def rollout_evaluate_fast(model, valid_dl, test_dl, train_xu, train_xi):
    valid_xu, valid_xi, valid_ranks = rollout(valid_dl, model, train_xu, train_xi)
    print(f"------- Valid MRR: {mrr(valid_ranks):.4f} Recall@10: {recall_at_k(valid_ranks, 10):.4f}")
    _u, _i, test_ranks = rollout(test_dl, model, valid_xu, valid_xi)
    print(f"=======  Test MRR: {mrr(test_ranks):.4f} Recall@10: {recall_at_k(test_ranks, 10):.4f}")


def rollout_evaluate(model, train_dl, valid_dl, test_dl):
    train_xu, train_xi, train_ranks = rollout(train_dl, model, *model.get_init_states())
    print(f"Train MRR: {mrr(train_ranks):.4f} Recall@10: {recall_at_k(train_ranks, 10):.4f}")
    valid_xu, valid_xi, valid_ranks = rollout(valid_dl, model, train_xu, train_xi)
    print(f"Valid MRR: {mrr(valid_ranks):.4f} Recall@10: {recall_at_k(valid_ranks, 10):.4f}")
    _u, _i, test_ranks = rollout(test_dl, model, valid_xu, valid_xi)
    print(f"Test MRR: {mrr(test_ranks):.4f} Recall@10: {recall_at_k(test_ranks, 10):.4f}")


def rollout(dl, model, last_xu, last_xi):
    model.eval()
    ranks = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dl, position=0):
            t, dt, adj, i2u_adj, u2i_adj, users, items = batch
            prop_user, prop_item, last_xu, last_xi = model.propagate_update(adj, dt, last_xu, last_xi, i2u_adj, u2i_adj)
            rs = compute_rank(model, prop_user, prop_item, users, items)
            ranks.extend(rs)
    return last_xu, last_xi, ranks


def compute_rank(model: CoPE, xu, xi, users, items):
    xu = torch.cat([xu, model.user_states], 1)
    xi = torch.cat([xi, model.item_states], 1)
    xu = F.embedding(users, xu)
    scores = model.compute_pairwise_scores(xu, xi)
    ranks = []
    for line, i in zip(scores, items):
        r = (line >= line[i]).sum().item()
        ranks.append(r)
    return ranks

