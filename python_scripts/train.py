"""Training loop. Run `python python_scripts/train.py` to train and save model.pt."""
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from data import MODEL_PATH, NUM_CLASSES, list_meshes
from features import build_datasets, hks_stats
from model import MiniDiffusionNet


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, data, optimizer=None, noise=0.0):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, count = 0.0, 0, 0
    order = random.sample(range(len(data)), len(data)) if train else range(len(data))
    with torch.set_grad_enabled(train):
        for i in order:
            d = data[i]
            x = d["x"] + noise * torch.randn_like(d["x"]) if train else d["x"]
            logits = model(x, d["evals"], d["evecs"], d["mass"], d["Fc"])
            loss = F.cross_entropy(logits, d["y"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(d["y"])
            correct += (logits.argmax(1) == d["y"]).sum().item()
            count += len(d["y"])
    return total_loss / count, correct / count


def save_checkpoint(model, history, t, mean, std, path=MODEL_PATH):
    """Weights plus the HKS times and normalization needed to run on new meshes."""
    torch.save({"model": model.state_dict(), "history": dict(history),
                "t": t.cpu(), "mean": mean.cpu(), "std": std.cpu()}, path)


def train(model, train_dev, test_dev, epochs=100, lr=1e-3, weight_decay=1e-4, noise=0.05, eval_every=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = defaultdict(list)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dev, optimizer, noise=noise)
        scheduler.step()
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)

        if epoch % eval_every == 0 or epoch == 1:
            te_loss, te_acc = run_epoch(model, test_dev)
            history["test_epoch"].append(epoch)
            history["test_loss"].append(te_loss)
            history["test_acc"].append(te_acc)
            print(f"epoch {epoch:3d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"test loss {te_loss:.3f} acc {te_acc:.3f} | {time.time() - t0:.0f}s")
    return history


if __name__ == "__main__":
    set_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_paths, test_paths = list_meshes()
    train_data, _, train_dev, test_dev, t = build_datasets(train_paths, test_paths, device)

    model = MiniDiffusionNet(len(t), NUM_CLASSES).to(device)
    history = train(model, train_dev, test_dev, epochs=100)

    save_checkpoint(model, history, t, *hks_stats(train_data))
    print(f"saved {MODEL_PATH}")
