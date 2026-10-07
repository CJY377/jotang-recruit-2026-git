# -*- coding: utf-8 -*-
"""Task 1: make_moons + PyTorch MLP 二分类全流程实验
固定随机种子，保证可复现。
"""
import os
import json
import time
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------- 全局设置 ----------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 中文字体
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cpu")


# ---------------- 数据 ----------------
def make_data(n=2000, noise=0.20, seed=SEED):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.astype(np.float32), y.astype(np.int64)


def split(X, y, seed=SEED):
    # 60% 训练 / 20% 验证 / 20% 测试
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def to_loader(X, y, batch_size=64, shuffle=False, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=batch_size, shuffle=shuffle, generator=g)


# ---------------- 模型 ----------------
ACTS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=(32,), act="relu", out_dim=2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), ACTS[act]()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------- 训练 ----------------
@torch.no_grad()
def evaluate(model, X, y, batch_size=256):
    model.eval()
    correct = 0
    loader = to_loader(X, y, batch_size=batch_size)
    for xb, yb in loader:
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
    return correct / len(y)


def train_model(X_tr, y_tr, X_val, y_val, *, hidden=(32,), act="relu",
                lr=0.01, optimizer="adam", batch_size=64, epochs=200,
                seed=SEED, weight_decay=0.0, log_every=10, verbose=True):
    torch.manual_seed(seed)
    model = MLP(hidden=hidden, act=act).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = {
        "adam": lambda ps: torch.optim.Adam(ps, lr=lr, weight_decay=weight_decay),
        "sgd": lambda ps: torch.optim.SGD(ps, lr=lr, weight_decay=weight_decay),
        "sgd_mom": lambda ps: torch.optim.SGD(ps, lr=lr, momentum=0.9, weight_decay=weight_decay),
    }[optimizer](model.parameters())

    tr_loader = to_loader(X_tr, y_tr, batch_size=batch_size, shuffle=True, seed=seed)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tot_loss, correct = 0.0, 0
        for xb, yb in tr_loader:
            opt.zero_grad()          # 1) 清空上一步残留梯度
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()          # 2) 反向传播，计算梯度
            opt.step()               # 3) 用梯度更新参数
            tot_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
        dt = time.perf_counter() - t0

        tr_loss, tr_acc = tot_loss / len(y_tr), correct / len(y_tr)
        # 验证集 loss
        model.eval()
        with torch.no_grad():
            vl = nn.functional.cross_entropy(model(torch.from_numpy(X_val)), torch.from_numpy(y_val)).item()
        va = evaluate(model, X_val, y_val)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        history["epoch_time"].append(dt)
        if verbose and (ep % log_every == 0 or ep == 1):
            print(f"  epoch {ep:3d} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val_loss {vl:.4f} acc {va:.4f} | {dt*1000:.1f} ms/epoch")

    return model, history


# ---------------- 画图工具 ----------------
CM = ListedColormap(["#e8746a", "#5b8db8"])  # 类0红 / 类1蓝（分类用色，非涨跌）


def plot_data(X_tr, y_tr, X_val, y_val, X_te, y_te, path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (X, y, t) in zip(axes[:3], [(X_tr, y_tr, "训练集 60%"), (X_val, y_val, "验证集 20%"), (X_te, y_te, "测试集 20%")]):
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CM, s=10, edgecolors="white", linewidths=0.2)
        ax.set_title(f"{t}（n={len(y)}）"); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax = axes[3]
    ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap=CM, s=8, alpha=0.35, label="train")
    ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=CM, s=18, marker="^", edgecolors="k", linewidths=0.4, label="val")
    ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap=CM, s=18, marker="s", edgecolors="k", linewidths=0.4, label="test")
    ax.legend(); ax.set_title("划分总览")
    fig.suptitle("make_moons 原始数据与数据集划分", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_curves(history, path, title="基线模型训练曲线"):
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["val_loss"], label="val")
    axes[0].set_title("Loss 曲线"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(ep, history["train_acc"], label="train")
    axes[1].plot(ep, history["val_acc"], label="val")
    axes[1].set_title("Accuracy 曲线"); axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[2].plot(ep, np.cumsum(history["epoch_time"]))
    axes[2].set_title("累计训练耗时 (s)"); axes[2].set_xlabel("epoch")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


@torch.no_grad()
def plot_decision_boundary(model, X, y, path, title="决策边界", extra=None):
    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    Z = model(grid).argmax(1).numpy().reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.contourf(xx, yy, Z, cmap=CM, alpha=0.25, levels=1)
    ax.contour(xx, yy, Z, colors="k", linewidths=1.0, levels=[0.5])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CM, s=12, edgecolors="white", linewidths=0.2)
    if extra is not None:
        Xe, ye, pred = extra
        ax.scatter(Xe[:, 0], Xe[:, 1], s=90, facecolors="none",
                   edgecolors=["#c0392b" if p != t else "#1e8449" for p, t in zip(pred, ye)],
                   linewidths=1.8, marker="o")
        ax.set_title(f"{title}\n（圈出的为模型判断错误的样本）")
    else:
        ax.set_title(title)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_confusion(y_true, y_pred, path, title="测试集混淆矩阵", normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["预测:类0", "预测:类1"])
    ax.set_yticks([0, 1], ["真实:类0", "真实:类1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)
    return cm


# ---------------- 对照实验 ----------------
def ablation(X_tr, y_tr, X_val, y_val, X_te, y_te, tag, configs, base_kwargs):
    """一次只改一个变量的对照实验，返回结果表"""
    rows = []
    curves = {}
    for name, override in configs:
        kwargs = dict(base_kwargs); kwargs.update(override)
        print(f"\n[{tag}] {name}: {override}")
        model, hist = train_model(X_tr, y_tr, X_val, y_val, verbose=False, **kwargs)
        te_acc = evaluate(model, X_te, y_te)
        best_val = max(hist["val_acc"])
        ep_best = int(np.argmax(hist["val_acc"])) + 1
        ep_time = float(np.mean(hist["epoch_time"]))
        rows.append(dict(name=name, params=count_params(model), best_val_acc=round(best_val, 4),
                         test_acc=round(te_acc, 4), best_epoch=ep_best, ms_per_epoch=round(ep_time * 1000, 1)))
        curves[name] = hist
        print(f"  参数量 {count_params(model)} | 最佳val_acc {best_val:.4f} (epoch {ep_best}) | test_acc {te_acc:.4f} | {ep_time*1000:.1f} ms/epoch")
    # 画 loss 对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, hist in curves.items():
        ep = np.arange(1, len(hist["train_loss"]) + 1)
        axes[0].plot(ep, hist["val_loss"], label=name)
        axes[1].plot(ep, hist["val_acc"], label=name)
    axes[0].set_title(f"{tag}：验证集 Loss 对比"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].set_title(f"{tag}：验证集 Accuracy 对比"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"ablation_{tag}.png"), dpi=130); plt.close(fig)
    return rows


# ---------------- 主流程 ----------------
def main():
    summary = {}

    # 1. 数据 + 划分
    X, y = make_data(n=2000, noise=0.20)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)
    print(f"数据: train={len(y_tr)} val={len(y_val)} test={len(y_te)}")
    plot_data(X_tr, y_tr, X_val, y_val, X_te, y_te, os.path.join(OUT, "01_data_split.png"))

    # 2. 基线模型: 2-32-2, ReLU, Adam(1e-2), bs=64
    print("\n===== 基线模型: hidden=(32,) ReLU Adam lr=0.01 bs=64 =====")
    base_kwargs = dict(hidden=(32,), act="relu", lr=0.01, optimizer="adam",
                       batch_size=64, epochs=200, log_every=40)
    model, hist = train_model(X_tr, y_tr, X_val, y_val, **base_kwargs)
    plot_curves(hist, os.path.join(OUT, "02_baseline_curves.png"))

    # 模型保存 / 加载并验证一致性
    ckpt = os.path.join(OUT, "mlp_baseline.pt")
    torch.save(model.state_dict(), ckpt)
    model2 = MLP(hidden=(32,), act="relu")
    model2.load_state_dict(torch.load(ckpt))
    model2.eval()
    acc_save = evaluate(model2, X_te, y_te)
    acc_orig = evaluate(model, X_te, y_te)
    assert abs(acc_save - acc_orig) < 1e-9, "保存/加载结果不一致!"
    print(f"模型保存/加载验证通过: test_acc={acc_save:.4f}")

    # 3. 测试 + 决策边界
    model.eval()
    with torch.no_grad():
        pred_te = model(torch.from_numpy(X_te)).argmax(1).numpy()
    te_acc = (pred_te == y_te).mean()
    print(f"基线测试集准确率: {te_acc:.4f}")
    plot_decision_boundary(model, X, y, os.path.join(OUT, "03_decision_boundary.png"),
                           "基线模型决策边界（全部 2000 样本）")
    summary["baseline"] = dict(test_acc=round(float(te_acc), 4), params=count_params(model),
                               ckpt=ckpt)

    # 4. 对照实验 A: 隐藏层宽度
    print("\n===== 对照实验 A: 隐藏层宽度 (其余固定) =====")
    rows_a = ablation(X_tr, y_tr, X_val, y_val, X_te, y_te, "width",
                      [("窄: h=4", dict(hidden=(4,))),
                       ("基线: h=32", dict(hidden=(32,))),
                       ("宽: h=128", dict(hidden=(128,))),
                       ("宽: h=512", dict(hidden=(512,)))], base_kwargs)
    summary["ablation_width"] = rows_a

    # 对照实验 B: 激活函数
    print("\n===== 对照实验 B: 激活函数 (其余固定) =====")
    rows_b = ablation(X_tr, y_tr, X_val, y_val, X_te, y_te, "activation",
                      [("ReLU", dict(act="relu")),
                       ("Tanh", dict(act="tanh")),
                       ("Sigmoid", dict(act="sigmoid"))], base_kwargs)
    summary["ablation_activation"] = rows_b

    # 对照实验 C: 学习率
    print("\n===== 对照实验 C: 学习率 (其余固定, SGD momentum=0.9) =====")
    sgd_kwargs = dict(base_kwargs); sgd_kwargs.update(optimizer="sgd_mom")
    rows_c = ablation(X_tr, y_tr, X_val, y_val, X_te, y_te, "lr",
                      [("lr=1.0(过大)", dict(lr=1.0, optimizer="sgd_mom")),
                       ("lr=0.1", dict(lr=0.1, optimizer="sgd_mom")),
                       ("lr=0.01", dict(lr=0.01, optimizer="sgd_mom")),
                       ("lr=0.0001(过小)", dict(lr=1e-4, optimizer="sgd_mom"))], sgd_kwargs)
    summary["ablation_lr"] = rows_c

    # 5. 混淆矩阵 + 错误样本
    cm = plot_confusion(y_te, pred_te, os.path.join(OUT, "05_confusion_matrix.png"))
    print(f"\n混淆矩阵:\n{cm}")
    err_idx = np.where(pred_te != y_te)[0]
    print(f"测试集错误样本数: {len(err_idx)}/{len(y_te)}")
    # 决策边界上标出错误样本
    plot_decision_boundary(model, X_te, y_te, os.path.join(OUT, "06_errors_on_boundary.png"),
                           "测试集错误样本分布", extra=(X_te[err_idx], y_te[err_idx], pred_te[err_idx]))
    err_detail = []
    for i in err_idx[:8]:
        with torch.no_grad():
            p = torch.softmax(model(torch.from_numpy(X_te[i:i + 1])), dim=1)[0].numpy()
        err_detail.append(dict(idx=int(i), x1=round(float(X_te[i, 0]), 3), x2=round(float(X_te[i, 1]), 3),
                               true=int(y_te[i]), pred=int(pred_te[i]),
                               p0=round(float(p[0]), 3), p1=round(float(p[1]), 3)))
        print(f"  样本{i}: x=({X_te[i,0]:.3f},{X_te[i,1]:.3f}) 真实={y_te[i]} 预测={pred_te[i]} "
              f"P(类0)={p[0]:.3f} P(类1)={p[1]:.3f}")
    summary["errors"] = err_detail

    # 6. 附加实验: 类别不均衡
    print("\n===== 附加实验: 类别不均衡 (类0:类1 = 1:20) =====")
    rng = np.random.default_rng(SEED)
    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]
    idx0_keep = rng.choice(idx0, size=len(idx1) // 20, replace=False)  # 类0只留 1/20
    imb_idx = np.concatenate([idx0_keep, idx1])
    rng.shuffle(imb_idx)
    Xi, yi = X[imb_idx], y[imb_idx]
    Xtr_i, Xva_i, Xte_i, ytr_i, yva_i, yte_i = split(Xi, yi)
    print(f"不均衡数据: train 类0={int((ytr_i==0).sum())} 类1={int((ytr_i==1).sum())}")
    m_i, h_i = train_model(Xtr_i, ytr_i, Xva_i, yva_i, verbose=False, **base_kwargs)
    with torch.no_grad():
        pi = m_i(torch.from_numpy(Xte_i)).argmax(1).numpy()
    acc_i = (pi == yte_i).mean()
    cm_i = confusion_matrix(yte_i, pi, labels=[0, 1])
    recall0 = cm_i[0, 0] / max(cm_i[0].sum(), 1)
    plot_confusion(yte_i, pi, os.path.join(OUT, "07_confusion_imbalanced.png"),
                   f"不均衡数据混淆矩阵\n(类0:类1≈1:20, acc={acc_i:.3f} 但类0召回={recall0:.3f})")
    plot_decision_boundary(m_i, X[::5], y[::5], os.path.join(OUT, "08_boundary_imbalanced.png"),
                           "不均衡模型的决策边界（边界向多数类一侧偏移）")
    summary["imbalanced"] = dict(acc=round(float(acc_i), 4), recall_minor=round(float(recall0), 4),
                                 cm=cm_i.tolist())
    print(f"  总体 acc={acc_i:.4f} | 少数类(类0)召回率={recall0:.4f}")

    # 7. 附加实验: 故意过拟合 (20 条训练样本 + 超大模型 + 无正则 + 2000 epochs)
    print("\n===== 附加实验: 故意过拟合 (20 样本 + 3x256 网络 + 2000 epochs) =====")
    idx_small = rng.choice(len(X_tr), size=20, replace=False)
    Xs, ys = X_tr[idx_small], y_tr[idx_small]
    big_kwargs = dict(hidden=(256, 256, 256), act="relu", lr=0.005, optimizer="adam",
                      batch_size=20, epochs=2000, log_every=500, weight_decay=0.0, seed=SEED)
    m_o, h_o = train_model(Xs, ys, X_val, y_val, verbose=True, **big_kwargs)
    plot_curves(h_o, os.path.join(OUT, "09_overfit_curves.png"),
                "过拟合实验：train_loss→0 而 val_loss 反升")
    with torch.no_grad():
        po = m_o(torch.from_numpy(X_te)).argmax(1).numpy()
    acc_o = (po == y_te).mean()
    print(f"  过拟合模型: 训练集 acc={h_o['train_acc'][-1]:.4f}, 测试集 acc={acc_o:.4f}")
    plot_decision_boundary(m_o, X, y, os.path.join(OUT, "10_boundary_overfit.png"),
                           f"过拟合模型决策边界（训练acc=100%, 测试acc={acc_o:.2%}）\n"
                           "边界出现贴合训练点的扭曲锯齿")
    summary["overfit"] = dict(train_acc=round(h_o["train_acc"][-1], 4),
                              val_acc_final=round(h_o["val_acc"][-1], 4),
                              test_acc=round(float(acc_o), 4),
                              val_loss_final=round(h_o["val_loss"][-1], 4))

    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n全部完成，结果已保存到 output/summary.json")


if __name__ == "__main__":
    main()
