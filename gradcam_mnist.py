"""
GradCAM on MNIST — standalone script
=====================================
Trains (or loads) a simple CNN on MNIST, then runs GradCAM and saves
a grid of visualisations to 'gradcam_results.png'.

Usage
-----
    python gradcam_mnist.py                 # train + visualise
    python gradcam_mnist.py --load          # load saved weights + visualise
    python gradcam_mnist.py --epochs 3      # train for 3 epochs

Requires: torch, torchvision, numpy, matplotlib
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 1. Network
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    Two-block ConvNet for MNIST.

    Input  : (B, 1, 28, 28)
    Block 1: Conv(1→16, 3×3, pad=1) → ReLU → MaxPool(2)  →  (B, 16, 14, 14)
    Block 2: Conv(16→32,3×3, pad=1) → ReLU → MaxPool(2)  →  (B, 32,  7,  7)
    Head   : Flatten → FC(128) → ReLU → FC(10)

    GradCAM target: self.conv2
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# 2. Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out  = model(images)
            loss = criterion(out, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# 3. GradCAM
# ---------------------------------------------------------------------------

def gradcam(model, input_tensor, target_layer, target_class=None):
    """
    Compute GradCAM for a single image.

    Steps
    -----
    1. Register forward hook  → capture feature maps A^k  (shape: C×h×w)
    2. Register backward hook → capture gradients dY^c/dA^k
    3. Forward pass  → get logits, choose target class
    4. Backward pass → gradients flow through target_layer hook
    5. Weights α_k   = mean over (h, w) of gradients      (shape: C)
    6. CAM           = Σ_k  α_k · A^k                     (shape: h×w)
    7. ReLU + normalise to [0, 1]
    8. Bilinear upsample to input H×W

    Parameters
    ----------
    model        : nn.Module — trained model
    input_tensor : Tensor   — shape (1, C, H, W)
    target_layer : nn.Module — layer to explain (e.g. model.conv2)
    target_class : int|None  — class index; None → argmax of output

    Returns
    -------
    cam_np     : np.ndarray  shape (H, W), values ∈ [0, 1]
    pred_class : int
    """
    _fmaps = {}
    _grads = {}

    def _fwd_hook(module, inp, out):
        _fmaps['x'] = out                  # (1, C, h, w)

    def _bwd_hook(module, grad_in, grad_out):
        _grads['x'] = grad_out[0]          # (1, C, h, w)

    h_fwd = target_layer.register_forward_hook(_fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        model.eval()

        # --- Step 3: Forward ---
        output     = model(input_tensor)           # (1, num_classes)
        pred_class = output.argmax(dim=1).item()
        if target_class is None:
            target_class = pred_class

        # --- Step 4: Backward ---
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        model.zero_grad()
        output.backward(gradient=one_hot)

        # --- Step 5: Gradient weights (global average pooling) ---
        grads = _grads['x']                        # (1, C, h, w)
        alpha = grads.mean(dim=[2, 3])             # (1, C)

        # --- Step 6: Weighted combination ---
        fmaps = _fmaps['x']                        # (1, C, h, w)
        cam   = (alpha.unsqueeze(-1).unsqueeze(-1) * fmaps).sum(dim=1)  # (1, h, w)

        # --- Step 7: ReLU + normalise ---
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # --- Step 8: Upsample ---
        H, W   = input_tensor.shape[-2:]
        cam    = F.interpolate(cam.unsqueeze(0), size=(H, W),
                               mode='bilinear', align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()    # (H, W)

    finally:
        h_fwd.remove()
        h_bwd.remove()

    return cam_np, pred_class


# ---------------------------------------------------------------------------
# 4. Visualisation helpers
# ---------------------------------------------------------------------------

def make_overlay(img_np, cam_np, alpha=0.5):
    """Blend a greyscale image and a jet-coloured CAM."""
    img_display = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    heatmap_rgb = cm.jet(cam_np)[:, :, :3]
    img_rgb     = np.stack([img_display] * 3, axis=-1)
    overlay     = np.clip(alpha * heatmap_rgb + (1 - alpha) * img_rgb, 0, 1)
    return img_display, cam_np, overlay


def plot_grid(model, dataset, indices, target_layer, device,
              target_class=None, save_path='gradcam_results.png'):
    """
    For each index in `indices`, plot: original | CAM | overlay.
    Saves to save_path.
    """
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img_tensor, true_label = dataset[idx]
        inp    = img_tensor.unsqueeze(0).to(device)
        cam_np, pred = gradcam(model, inp, target_layer, target_class)

        img_np         = img_tensor.squeeze().numpy()
        img_d, cam_d, ov = make_overlay(img_np, cam_np)

        axes[row, 0].imshow(img_d, cmap='gray')
        axes[row, 0].set_title(f'[{idx}]  label={true_label}')

        im = axes[row, 1].imshow(cam_d, cmap='jet', vmin=0, vmax=1)
        axes[row, 1].set_title(f'GradCAM  pred={pred}')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

        axes[row, 2].imshow(ov)
        axes[row, 2].set_title('Overlay')

        for ax in axes[row]:
            ax.axis('off')

    plt.suptitle('GradCAM on MNIST', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='GradCAM on MNIST')
    parser.add_argument('--epochs',     type=int,  default=5,
                        help='number of training epochs (default: 5)')
    parser.add_argument('--load',       action='store_true',
                        help='load weights from mnist_cnn.pth instead of training')
    parser.add_argument('--weight-path', type=str, default='mnist_cnn.pth',
                        help='path to save/load model weights')
    parser.add_argument('--output',     type=str, default='gradcam_results.png',
                        help='output image path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- Data ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0)

    # --- Model ---
    model = SimpleCNN().to(device)
    weight_path = Path(args.weight_path)

    if args.load or weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f'Loaded weights from {weight_path}')
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_ld, optimizer, criterion, device)
            va_loss, va_acc = evaluate(model, test_ld, criterion, device)
            print(f'Epoch {ep}/{args.epochs}  '
                  f'train loss {tr_loss:.4f} acc {tr_acc:.4f}  |  '
                  f'test  loss {va_loss:.4f} acc {va_acc:.4f}')
        torch.save(model.state_dict(), weight_path)
        print(f'Weights saved → {weight_path}')

    _, test_acc = evaluate(model, test_ld, nn.CrossEntropyLoss(), device)
    print(f'Test accuracy: {test_acc*100:.2f}%')

    # --- Pick one image per digit ---
    one_per_class = []
    seen = set()
    for i, (_, lbl) in enumerate(test_ds):
        if lbl not in seen:
            one_per_class.append(i)
            seen.add(lbl)
        if len(seen) == 10:
            break

    plot_grid(model, test_ds, one_per_class, model.relu2, device,
              save_path=args.output)


if __name__ == '__main__':
    main()
