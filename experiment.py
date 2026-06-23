import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "缺陷预测有关数据集"
RESULT_DIR = ROOT / "experiment" / "results"


@dataclass
class ProjectData:
    group: str
    name: str
    path: Path
    x: np.ndarray
    y: np.ndarray

    @property
    def feature_dim(self):
        return self.x.shape[1]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def label_to_binary(values):
    s = pd.Series(values).astype(str).str.upper()
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0)
    return ((s.isin(["TRUE", "Y", "YES", "1"])) | (numeric > 0)).astype(int).to_numpy()


def load_projects(data_root: Path):
    projects = []
    for path in sorted(data_root.rglob("*.csv")):
        df = pd.read_csv(path)
        x = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y = label_to_binary(df.iloc[:, -1])
        projects.append(ProjectData(path.parent.name, path.stem, path, x, y.astype(np.int64)))
    if len(projects) != 28:
        raise RuntimeError(f"Expected 28 csv files, found {len(projects)} in {data_root}")
    return projects


def profile_features(x):
    x = np.asarray(x, dtype=np.float64)
    stats = []
    for func in (np.mean, np.std, np.min, np.max, np.median):
        v = func(x, axis=0)
        stats.extend([np.nanmean(v), np.nanstd(v), np.nanmin(v), np.nanmax(v)])
    stats.extend([x.shape[0], x.shape[1]])
    arr = np.asarray(stats, dtype=np.float64)
    arr[~np.isfinite(arr)] = 0
    return arr


def source_weights(target, sources, enabled=True):
    if not enabled:
        return {s.name: 1.0 / len(sources) for s in sources}
    profs = np.vstack([profile_features(p.x) for p in [target] + sources])
    scaler = StandardScaler()
    profs = scaler.fit_transform(profs)
    t = profs[0]
    sims = []
    for i, _src in enumerate(sources, start=1):
        d = np.linalg.norm(t - profs[i])
        sims.append(math.exp(-d / max(len(t), 1) ** 0.5))
    sims = np.asarray(sims, dtype=np.float64)
    sims = sims / sims.sum() if sims.sum() > 0 else np.ones_like(sims) / len(sims)
    return {s.name: float(w) for s, w in zip(sources, sims)}


def split_target(project, seed):
    if len(np.unique(project.y)) < 2 or len(project.y) < 10:
        return project.x, project.y, project.x, project.y
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_idx, test_idx = next(splitter.split(project.x, project.y))
    return project.x[train_idx], project.y[train_idx], project.x[test_idx], project.y[test_idx]


def scale_by_project(projects, target_train=None, target_test=None):
    scaled = {}
    for p in projects:
        scaler = StandardScaler()
        scaled[p.name] = scaler.fit_transform(p.x).astype(np.float32)
    t_train = t_test = None
    if target_train is not None:
        scaler = StandardScaler()
        t_train = scaler.fit_transform(target_train).astype(np.float32)
        t_test = scaler.transform(target_test).astype(np.float32)
    return scaled, t_train, t_test


class EncoderAE(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        hidden = max(16, min(96, in_dim * 2))
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        return z, rec


class MultiSourceModel(nn.Module):
    def __init__(self, dims, latent_dim):
        super().__init__()
        self.name_to_key = {name: f"p{i}" for i, name in enumerate(dims)}
        self.encoders = nn.ModuleDict({self.name_to_key[name]: EncoderAE(dim, latent_dim) for name, dim in dims.items()})
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, max(8, latent_dim // 2)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(max(8, latent_dim // 2), 1),
        )

    def encode(self, name, x):
        return self.encoders[self.name_to_key[name]](x)

    def logits_from_z(self, z):
        return self.classifier(z).squeeze(-1)


def coral_loss(source, target):
    if source.shape[0] < 2 or target.shape[0] < 2:
        return source.new_tensor(0.0)
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    cs = source.t().mm(source) / (source.shape[0] - 1)
    ct = target.t().mm(target) / (target.shape[0] - 1)
    return ((cs - ct) ** 2).mean()


def focal_bce(logits, y, alpha, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(y == 1, p, 1 - p)
    alpha_t = torch.where(y == 1, alpha, 1 - alpha)
    return (alpha_t * (1 - pt).pow(gamma) * bce).mean()


def make_loader(x, y=None, batch_size=64, seed=42):
    gen = torch.Generator().manual_seed(seed)
    xt = torch.tensor(x, dtype=torch.float32)
    if y is None:
        ds = TensorDataset(xt)
    else:
        ds = TensorDataset(xt, torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=gen)


def cycle_next(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def train_deep(target, sources, args, variant):
    set_seed(args.seed)
    scaled, _, _ = scale_by_project([target] + sources)
    dims = {target.name: target.feature_dim}
    dims.update({s.name: s.feature_dim for s in sources})
    model = MultiSourceModel(dims, args.latent_dim)
    weights = source_weights(target, sources, enabled=(variant != "no_similarity"))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    target_loader = make_loader(scaled[target.name], None, args.batch_size, args.seed)
    source_loaders = {
        s.name: make_loader(scaled[s.name], s.y, args.batch_size, args.seed + i + 7)
        for i, s in enumerate(sources)
    }
    target_iter = iter(target_loader)
    source_iters = {name: iter(loader) for name, loader in source_loaders.items()}
    positives = sum(int(s.y.sum()) for s in sources)
    total = sum(len(s.y) for s in sources)
    alpha = torch.tensor(max(0.15, min(0.85, 1 - positives / max(total, 1))), dtype=torch.float32)

    max_steps = max(1, args.steps_per_epoch)
    for _epoch in range(args.epochs):
        model.train()
        for _ in range(max_steps):
            (tb,), target_iter = cycle_next(target_iter, target_loader)
            opt.zero_grad()
            zt, rt = model.encode(target.name, tb)
            loss = 0.05 * F.mse_loss(rt, tb)
            for s in sources:
                (xb, yb), source_iters[s.name] = cycle_next(source_iters[s.name], source_loaders[s.name])
                zs, rs = model.encode(s.name, xb)
                logits = model.logits_from_z(zs)
                if variant == "no_imbalance":
                    cls = F.binary_cross_entropy_with_logits(logits, yb)
                else:
                    cls = focal_bce(logits, yb, alpha)
                rec = F.mse_loss(rs, xb)
                align = coral_loss(zs, zt.detach()) if variant != "no_alignment" else zs.new_tensor(0.0)
                loss = loss + weights[s.name] * (cls + 0.05 * rec + args.align_lambda * align)
            loss.backward()
            opt.step()
    return model


def predict_deep(model, target, x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(target.x).astype(np.float32)
    # Refit with full target features; the encoder has only seen target reconstruction, not labels.
    # For evaluation, transform the supplied array using target-wide feature scale.
    scaler = StandardScaler().fit(target.x)
    x_scaled = scaler.transform(x).astype(np.float32)
    model.eval()
    with torch.no_grad():
        z, _ = model.encode(target.name, torch.tensor(x_scaled, dtype=torch.float32))
        prob = torch.sigmoid(model.logits_from_z(z)).cpu().numpy()
    return prob


def metrics_from_prob(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=labels).ravel()
    pd_ = tp / (tp + fn) if (tp + fn) else 0.0
    pf = fp / (fp + tn) if (fp + tn) else 0.0
    gm = math.sqrt(pd_ * (1 - pf)) if pd_ >= 0 and pf <= 1 else 0.0
    f1 = f1_score(y_true, pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
    except ValueError:
        auc = np.nan
    return {"Pd": pd_, "Pf": pf, "GM": gm, "AUC": auc, "F1": f1, "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def run_classical(target, sources, seed, method):
    common_dim = min([target.feature_dim] + [s.feature_dim for s in sources])
    xs = []
    ys = []
    for s in sources:
        xs.append(s.x[:, :common_dim])
        ys.append(s.y)
    x_train = np.vstack(xs)
    y_train = np.concatenate(ys)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(target.x[:, :common_dim])
    if method == "LR_common":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    else:
        clf = RandomForestClassifier(n_estimators=160, class_weight="balanced", random_state=seed, n_jobs=1)
    clf.fit(x_train, y_train)
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(x_test)[:, 1]
    else:
        prob = clf.decision_function(x_test)
    return prob


def run_wpdp(target, seed):
    x_train, y_train, x_test, y_test = split_target(target, seed)
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(scaler.transform(x_train), y_train)
    prob = clf.predict_proba(scaler.transform(x_test))[:, 1]
    return y_test, prob


def run(args):
    set_seed(args.seed)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    projects = load_projects(args.data_root)
    if args.quick:
        target_names = ["cm1", "Apache", "eclipseJDTCore", "ant-1.7"]
        projects_to_run = [p for p in projects if p.name in target_names]
        args.epochs = min(args.epochs, 12)
        args.steps_per_epoch = min(args.steps_per_epoch, 8)
    elif args.targets != "all":
        names = {x.strip() for x in args.targets.split(",")}
        projects_to_run = [p for p in projects if p.name in names]
    else:
        projects_to_run = projects

    deep_variants = ["SMDP_full", "no_similarity", "no_alignment", "no_imbalance"]
    rows = []
    weight_rows = []
    for target in projects_to_run:
        sources = [p for p in projects if p.group != target.group]
        weights = source_weights(target, sources, enabled=True)
        for name, w in weights.items():
            weight_rows.append({"target": target.name, "source": name, "weight": w})
        for variant in deep_variants:
            model = train_deep(target, sources, args, variant)
            prob = predict_deep(model, target, target.x)
            m = metrics_from_prob(target.y, prob)
            rows.append({"target": target.name, "group": target.group, "method": variant, **m})
        for method in ["LR_common", "RF_common"]:
            prob = run_classical(target, sources, args.seed, method)
            m = metrics_from_prob(target.y, prob)
            rows.append({"target": target.name, "group": target.group, "method": method, **m})
        y_wp, prob_wp = run_wpdp(target, args.seed)
        rows.append({"target": target.name, "group": target.group, "method": "WPDP_LR_upper", **metrics_from_prob(y_wp, prob_wp)})
        print(f"finished {target.name}")

    metrics = pd.DataFrame(rows)
    suffix = "quick" if args.quick else "full"
    metrics_path = RESULT_DIR / f"metrics_{suffix}.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(weight_rows).to_csv(RESULT_DIR / f"source_weights_{suffix}.csv", index=False, encoding="utf-8-sig")

    summary = metrics.groupby("method")[["Pd", "Pf", "GM", "AUC", "F1"]].agg(["mean", "std"]).round(4)
    summary.to_csv(RESULT_DIR / f"summary_{suffix}.csv", encoding="utf-8-sig")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (RESULT_DIR / f"run_config_{suffix}.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--targets", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--steps-per-epoch", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--align-lambda", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
