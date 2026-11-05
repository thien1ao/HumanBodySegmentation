import torch
from data import NUM_CLASSES, compute_areas


@torch.no_grad()
def predict(model, dd):
    model.eval()
    logits = model(dd["x"], dd["evals"], dd["evecs"], dd["mass"], dd["Fc"])
    return logits.argmax(1).cpu()


def confusion_matrix(model, data, data_dev, weight_by_area=True):
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.float64)
    for d, dd in zip(data, data_dev):
        pred, y = predict(model, dd), d["labels"]
        w = compute_areas(d["V"], d["Fc"]) if weight_by_area else torch.ones(len(y), dtype=torch.float64)
        cm.index_put_((y, pred), w, accumulate=True)
    return cm


def iou_from_confusion(cm):
    tp = cm.diag()
    return tp / (cm.sum(0) + cm.sum(1) - tp).clamp(min=1e-12)
