"""
PyTorch component: a small CNN that classifies Sentinel-2 image patches as
flooded / not-flooded, complementing the XGBoost terrain+rainfall model.

HONEST STATUS: this defines and trains a real, working CNN architecture, but
this sandbox cannot download real Sentinel-2 imagery (no network access to
Earth Engine). It trains here on synthetic patches (bright/uniform = water-like
signature vs textured = land-like signature) purely so the training loop and
architecture are proven to run end-to-end. To make it real:

    1. Use gee_integration.get_latest_sentinel2_ndwi() to pull real NDWI
       patches for labeled flood/no-flood locations
    2. Replace `make_synthetic_batch()` below with a real DataLoader over
       those downloaded patches
    3. Everything else (model, training loop, inference function) is
       unchanged

In production this model's output (a per-tile flood probability from actual
satellite imagery) would be fused with the XGBoost terrain+rainfall score —
XGBoost predicts risk before/during a storm from terrain+rainfall, this CNN
confirms actual flooding from imagery once satellite passes are available.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class FloodExtentCNN(nn.Module):
    """Small CNN for binary flooded/not-flooded classification of a 32x32
    multi-band (NDWI + VV SAR backscatter) satellite image patch."""

    def __init__(self, in_channels=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),            # 16 -> 8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),    # -> 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_synthetic_batch(batch_size=32, patch_size=32):
    """
    NOT real satellite data — synthetic stand-in so the training loop is
    provably runnable in this sandbox. Water-like patches: high, uniform
    NDWI + low, uniform SAR backscatter. Land-like: noisy/textured in both
    bands. Replace with real downloaded Sentinel-1/2 patches for production.
    """
    half = batch_size // 2
    water = torch.stack([
        torch.full((patch_size, patch_size), 0.7) + torch.randn(patch_size, patch_size) * 0.05,
        torch.full((patch_size, patch_size), 0.2) + torch.randn(patch_size, patch_size) * 0.05,
    ], dim=0).unsqueeze(0).repeat(half, 1, 1, 1)
    land = torch.stack([
        torch.rand(patch_size, patch_size) * 0.6,
        torch.rand(patch_size, patch_size) * 0.6,
    ], dim=0).unsqueeze(0).repeat(batch_size - half, 1, 1, 1)
    x = torch.cat([water, land], dim=0)
    y = torch.cat([torch.ones(half, 1), torch.zeros(batch_size - half, 1)], dim=0)
    perm = torch.randperm(batch_size)
    return x[perm], y[perm]


def train_demo(epochs=8, steps_per_epoch=20):
    model = FloodExtentCNN()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            x, y = make_synthetic_batch()
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}/{epochs} — avg loss {total_loss/steps_per_epoch:.4f}")

    torch.save(model.state_dict(), "model/artifacts/flood_extent_cnn.pt")
    return model


@torch.no_grad()
def predict_patch(model, ndwi_patch, sar_patch):
    """Run inference on a single real (or synthetic) 32x32 NDWI+SAR patch pair."""
    model.eval()
    x = torch.stack([torch.tensor(ndwi_patch, dtype=torch.float32),
                      torch.tensor(sar_patch, dtype=torch.float32)], dim=0).unsqueeze(0)
    prob = torch.sigmoid(model(x)).item()
    return prob


if __name__ == "__main__":
    import os
    os.makedirs("model/artifacts", exist_ok=True)
    m = train_demo()
    x, y = make_synthetic_batch(batch_size=4)
    with torch.no_grad():
        preds = torch.sigmoid(m(x))
    print("Sample predictions (should roughly match labels):")
    for p, label in zip(preds, y):
        print(f"  predicted flood prob={p.item():.3f}  actual label={int(label.item())}")
