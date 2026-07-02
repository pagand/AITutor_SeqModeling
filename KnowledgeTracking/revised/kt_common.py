"""
kt_common.py  —  Shared setup (v2: adds Brier, train/test gap, cold-start,
                  calibration-free variant). Imported by all runners.
"""
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from collections import defaultdict

EPOCHS = 12

# ---------- data ----------
df = pd.read_pickle("data/KT_logs_annotated.pkl")
# Reconcile to the 89 analyzable learners (recruited N=89): exclude learners
# with fewer than 10 logged interactions, which are too short to estimate a
# mastery trajectory (standard minimum-sequence-length filter in KT).
MIN_INTERACTIONS = 10
_counts = df.groupby('username').size()
_keep = _counts[_counts >= MIN_INTERACTIONS].index
df = df[df['username'].isin(_keep)].reset_index(drop=True)
assert df['username'].nunique() == 89, f"expected 89 users, got {df['username'].nunique()}"

skills_dict = pickle.load(open("data/Skill_hirereachy.pkl", "rb"))
valid_users = df["username"].unique()
user2idx = {u: i for i, u in enumerate(valid_users)}
all_skills = set()
for s_list in df['skill']:
    for s in s_list:
        all_skills.add(s)
skill2idx = {s: i for i, s in enumerate(sorted(all_skills))}
idx2skill = {i: s for s, i in skill2idx.items()}
hierarchy_edges = []
for child, data in skills_dict.items():
    for p in data[1]:
        if child in skill2idx and p in skill2idx:
            hierarchy_edges.append((skill2idx[child], skill2idx[p]))
parent_of = defaultdict(list)
for c, p in hierarchy_edges:
    parent_of[c].append(p)
edge_index = {(c, p): i for i, (c, p) in enumerate(hierarchy_edges)}
num_edges = max(1, len(hierarchy_edges))
num_users = len(user2idx)
num_skills = len(skill2idx)
df = df.sort_values(["username", "time"]).reset_index(drop=True)
base_rate = df['correct'].mean()

N_BUCKETS = 4
def attempt_bucket(n):
    if n <= 1:  return 0
    if n <= 3:  return 1
    if n <= 10: return 2
    return 3

# ---------- build sequences ONCE ----------
ALL_SEQS = defaultdict(list)
for uname, slist, corr in zip(df['username'].values, df['skill'].values, df['correct'].values):
    sidx = next((skill2idx[s] for s in slist if s in skill2idx), None)
    if sidx is not None:
        ALL_SEQS[user2idx[uname]].append((sidx, float(corr)))

ALL_DKT = {}   # u -> (X, target_skill, target_y, target_bucket, target_position)
for u, seq in ALL_SEQS.items():
    if len(seq) < 2:
        continue
    skills = [s for s, _ in seq]; corr = [int(c) for _, c in seq]
    seen = defaultdict(int); buckets = []
    for sk in skills:
        seen[sk] += 1; buckets.append(attempt_bucket(seen[sk]))
    inp = [skills[i] + num_skills * corr[i] for i in range(len(skills) - 1)]
    positions = list(range(2, len(skills) + 1))   # target j is absolute position j+2 (1-based)
    ALL_DKT[u] = (torch.tensor(inp, dtype=torch.long),
                  torch.tensor(skills[1:], dtype=torch.long),
                  torch.tensor(corr[1:], dtype=torch.float32),
                  buckets[1:], positions)

print(f"[kt_common] Users {num_users} | Skills {num_skills} | "
      f"Interactions {len(df)} | Base {base_rate:.4f} | Edges {len(hierarchy_edges)}")

# ---------- metric helpers ----------
def safe_auc(t, p):
    return roc_auc_score(t, p) if len(set(t)) > 1 else float('nan')

def brier(t, p):
    t = np.asarray(t, float); p = np.asarray(p, float)
    return float(np.mean((p - t) ** 2))

def ece(t, p, n_bins=10):
    t = np.asarray(t, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if m.sum():
            e += (m.sum() / len(p)) * abs(p[m].mean() - t[m].mean())
    return float(e)

def dyn_f1(t, p):
    best = 0.0
    for thr in np.arange(0.10, 0.91, 0.05):
        best = max(best, f1_score(t, [1 if x > thr else 0 for x in p], zero_division=0))
    return best

def coldstart_auc(t, p, pos, N):
    t = np.asarray(t); p = np.asarray(p); pos = np.asarray(pos)
    m = pos <= N
    return safe_auc(t[m].tolist(), p[m].tolist()) if m.sum() and len(set(t[m].tolist())) > 1 else float('nan')

def summarize(test, train):
    tp, tt, tb, tpos = test
    aucv = safe_auc(tt, tp)
    out = {
        'auc': aucv,
        'pr': average_precision_score(tt, tp),
        'f1': dyn_f1(tt, tp),
        'ece': ece(tt, tp),
        'brier': brier(tt, tp),
        'cs3': coldstart_auc(tt, tp, tpos, 3),
        'cs5': coldstart_auc(tt, tp, tpos, 5),
    }
    if train is not None:
        trp, trt = train
        tr_auc = safe_auc(trt, trp)
        out['train_auc'] = tr_auc
        out['gap'] = (tr_auc - aucv) if not (np.isnan(tr_auc) or np.isnan(aucv)) else float('nan')
    return out

# ---------- models ----------
class DKT(nn.Module):
    def __init__(self, n_skills, hidden=32):
        super().__init__()
        self.emb = nn.Embedding(2 * n_skills, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_skills)
    def forward(self, x):
        return torch.sigmoid(self.fc(self.lstm(self.emb(x))[0]))

class UnifiedBKT(nn.Module):
    def __init__(self, n_skills, n_users, use_irt=False, use_diff=False,
                 use_transfer=False, g_init=-1.5, free_slip=False):
        super().__init__()
        self.free_slip = free_slip
        self.skill_L0 = nn.Parameter(torch.randn(n_skills) * 0.1)
        self.skill_T  = nn.Parameter(torch.randn(n_skills) * 0.1)
        self.skill_G  = nn.Parameter(torch.randn(n_skills) * 0.1 + g_init)
        self.skill_S  = nn.Parameter(torch.randn(n_skills) * 0.1 - 1.5)
        if use_irt:
            self.user_shift_L0 = nn.Parameter(torch.zeros(n_users))
        else:
            self.register_buffer("user_shift_L0", torch.zeros(n_users))
        if use_diff:
            self.diff_G = nn.Parameter(torch.zeros(N_BUCKETS))
            self.diff_S = nn.Parameter(torch.zeros(N_BUCKETS))
        else:
            self.register_buffer("diff_G", torch.zeros(N_BUCKETS))
            self.register_buffer("diff_S", torch.zeros(N_BUCKETS))
        if use_transfer:
            self.transfer_alpha = nn.Parameter(torch.tensor(-1.0))
            self.transfer_rho   = nn.Parameter(torch.zeros(num_edges))
        else:
            self.register_buffer("transfer_alpha", torch.tensor(-1e4))
            self.register_buffer("transfer_rho", torch.zeros(num_edges))
        self.use_diff = use_diff; self.use_transfer = use_transfer
    def _scap(self):
        return (1 - 1e-4) if self.free_slip else 0.4
    def base_L0(self):
        return torch.clamp(torch.sigmoid(self.skill_L0.unsqueeze(0) + self.user_shift_L0.unsqueeze(1)), 1e-4, 1 - 1e-4)
    def skill_T_prob(self):
        return torch.clamp(torch.sigmoid(self.skill_T), 1e-4, 1 - 1e-4)
    def gs_for(self, s, b):
        G = torch.clamp(torch.sigmoid(self.skill_G[s] + self.diff_G[b]), 1e-4, 0.4)
        S = torch.clamp(torch.sigmoid(self.skill_S[s] + self.diff_S[b]), 1e-4, self._scap())
        return G, S

def flags(model_type):
    base_em = ("em", "em_diff", "em_transfer", "em_full", "em_cal")
    return dict(
        use_adam = model_type != "vanilla",
        use_hier = model_type in ("adam_hier", "mle", "map") + base_em,
        use_irt  = model_type in ("mle", "map") + base_em,
        use_map  = model_type in ("map",) + base_em,
        use_em   = model_type in base_em,
        use_diff = model_type in ("em_diff", "em_full"),
        use_tr   = model_type in ("em_transfer", "em_full"),
        g_init   = -2.5 if model_type == "em_cal" else -1.5,
        free_slip = model_type == "em_cal",
    )

# ---------- BKT forward over sequences ----------
def bkt_run(model, f, seqs, train_mode):
    L0_all = model.base_L0(); T_all = model.skill_T_prob()
    if not f['use_diff']:
        G_static = torch.clamp(torch.sigmoid(model.skill_G), 1e-4, 0.4)
        S_static = torch.clamp(torch.sigmoid(model.skill_S), 1e-4, model._scap())
    alpha = torch.sigmoid(model.transfer_alpha); rho = torch.sigmoid(model.transfer_rho)
    preds, trues, bks, pos = [], [], [], []
    for u, seq in seqs.items():
        if not seq: continue
        cur = {}; visits = defaultdict(int); step = 0
        for s_idx, y in seq:
            step += 1; visits[s_idx] += 1; b = attempt_bucket(visits[s_idx])
            if s_idx not in cur:
                base = L0_all[u, s_idx]
                if f['use_tr'] and parent_of.get(s_idx):
                    seen_p = [p for p in parent_of[s_idx] if p in cur]
                    if seen_p:
                        p_best = max(seen_p, key=lambda p: float(cur[p]))
                        ridx = edge_index[(s_idx, p_best)]
                        if train_mode:
                            base = base + alpha * rho[ridx] * torch.relu(cur[p_best] - base)
                        else:
                            base = base + float(alpha) * float(rho[ridx]) * max(0.0, float(cur[p_best]) - float(base))
                cur[s_idx] = base
            p_l = cur[s_idx]
            if f['use_diff']:
                G, S = model.gs_for(s_idx, b)
                if not train_mode: G, S = float(G), float(S)
            else:
                G = G_static[s_idx] if train_mode else float(G_static[s_idx])
                S = S_static[s_idx] if train_mode else float(S_static[s_idx])
            T = T_all[s_idx] if train_mode else float(T_all[s_idx])
            p_c = p_l * (1 - S) + (1 - p_l) * G
            preds.append(p_c); trues.append(y); bks.append(b); pos.append(step)
            p_c_safe = torch.clamp(p_c, 1e-4, 1 - 1e-4) if train_mode else max(1e-4, min(1 - 1e-4, p_c))
            p_l_obs = (p_l * (1 - S)) / p_c_safe if y == 1.0 else (p_l * S) / (1 - p_c_safe)
            if f['use_em'] and train_mode: p_l_obs = p_l_obs.detach()
            cur[s_idx] = p_l_obs + (1 - p_l_obs) * T
    return preds, trues, bks, pos

# ---------- train + eval (returns metric dict) ----------
def train_and_eval(train_uidx, test_uidx, model_type, fold_seed=42):
    torch.manual_seed(fold_seed); np.random.seed(fold_seed)
    if model_type == "dkt":
        tr = [ALL_DKT[u] for u in train_uidx if u in ALL_DKT]
        te = [ALL_DKT[u] for u in test_uidx if u in ALL_DKT]
        model = DKT(num_skills); opt = optim.Adam(model.parameters(), lr=0.01); crit = nn.BCELoss()
        model.train()
        for _ in range(EPOCHS):
            opt.zero_grad(); total = None
            for X, S, Y, _, _ in tr:
                pred = torch.gather(model(X.unsqueeze(0)).squeeze(0), 1, S.unsqueeze(1)).squeeze(1)
                li = crit(pred, Y); total = li if total is None else total + li
            if total is not None: total.backward(); opt.step()
        def ev(data):
            P, Tt, B, Po = [], [], [], []
            with torch.no_grad():
                for X, S, Y, Bk, Ps in data:
                    p = torch.gather(model(X.unsqueeze(0)).squeeze(0), 1, S.unsqueeze(1)).squeeze(1).numpy()
                    P.extend(p.tolist()); Tt.extend(Y.numpy().tolist()); B.extend(Bk); Po.extend(Ps)
            return P, Tt, B, Po
        model.eval()
        test = ev(te); trp, trt, _, _ = ev(tr)
        npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        f = flags(model_type)
        model = UnifiedBKT(num_skills, num_users, f['use_irt'], f['use_diff'], f['use_tr'],
                           g_init=f['g_init'], free_slip=f['free_slip'])
        opt = (optim.Adam(model.parameters(), lr=0.02) if f['use_adam'] else optim.SGD(model.parameters(), lr=0.1))
        crit = nn.BCELoss()
        train_seqs = {u: ALL_SEQS[u] for u in train_uidx if u in ALL_SEQS}
        test_seqs  = {u: ALL_SEQS[u] for u in test_uidx if u in ALL_SEQS}
        model.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            tp, tt, _, _ = bkt_run(model, f, train_seqs, True)
            loss = crit(torch.stack(tp), torch.tensor(tt, dtype=torch.float32))
            if f['use_map']:
                loss = loss + 0.05 * torch.mean(model.user_shift_L0 ** 2)
            if f['use_hier'] and hierarchy_edges:
                sL0 = torch.sigmoid(model.skill_L0)
                loss = loss + 0.5 * sum(torch.relu(sL0[c] - sL0[p] + 0.05) for c, p in hierarchy_edges)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            test = bkt_run(model, f, test_seqs, False)
            trp, trt, _, _ = bkt_run(model, f, train_seqs, False)
        npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    m = summarize(test, (trp, trt))
    m['params'] = npar
    return m
