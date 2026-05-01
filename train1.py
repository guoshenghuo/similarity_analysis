import torch
import os
import sys
import time
import copy
import numpy as np
import pandas as pd
import re
from PIL import Image
from torchvision import transforms as pth_transforms
import hubconf
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load models
dinov2_vits14 = hubconf.dinov2_vits14()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14 = dinov2_vits14.to(device)



# Parameters
patch_size = 14
n_components = 32
MODEL_CHECKPOINT_PATH = "best_neural_network_model.pt"

# ==================== Utility Functions ====================

def adjust_image_size(img, patch_size=14):
    """Crop image to be divisible by patch_size"""
    w, h = img.size
    new_w = w - (w % patch_size)
    new_h = h - (h % patch_size)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    img = img.crop((left, top, right, bottom))
    return img

def fill_nan_with_interpolation(arr):
    """Fill NaN values with interpolation"""
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    arr_filled = arr.copy()
    if arr_filled.ndim == 2:
        for i in range(arr_filled.shape[0]):
            row = arr_filled[i]
            nan_mask = np.isnan(row)
            if np.any(nan_mask):
                valid_idx = np.where(~nan_mask)[0]
                if len(valid_idx) == 0:
                    arr_filled[i, :] = 0.0
                else:
                    interp_vals = np.interp(np.where(nan_mask)[0], valid_idx, row[valid_idx])
                    arr_filled[i, nan_mask] = interp_vals
    else:
        flat = arr_filled.flatten()
        nan_mask = np.isnan(flat)
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) > 0:
            interp_vals = np.interp(np.where(nan_mask)[0], valid_idx, flat[valid_idx])
            flat[nan_mask] = interp_vals
            arr_filled = flat.reshape(arr_filled.shape)
        else:
            arr_filled = np.zeros_like(arr_filled)
    arr_filled = np.nan_to_num(arr_filled, nan=0.0, posinf=1e3, neginf=-1e3)
    return arr_filled

def extract_features(image_paths, return_patches=False):
    """Extract DINOv2 features from images"""
    features_list = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            img = adjust_image_size(img, patch_size=patch_size)
            transform = pth_transforms.Compose([
                pth_transforms.Resize(img.size),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            img_tensor = transform(img)[:3].unsqueeze(0).to(device)
            with torch.no_grad():
                if return_patches:
                    features = dinov2_vits14(img_tensor, return_patches=True)[0]
                else:
                    features = dinov2_vits14(img_tensor)
            features_np = features.detach().cpu().numpy()
            if np.isnan(features_np).any() or np.isinf(features_np).any():
                features_np = fill_nan_with_interpolation(features_np)
            features_list.append(features_np)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            features_list.append(np.zeros((1, 384)))
    return features_list

def load_similarity_pairs(txt_path='./similarity.txt'):
    """Load pairs from mixed delimiter file"""
    pairs = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for line_num, line in enumerate(lines, 1):
        parts = re.split(r'[\t,]', line)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) != 3:
            print(f"Warning: Line {line_num} invalid format. Skipping.")
            continue
        
        img1, img2, sim_str = parts
        
        if img1.startswith('./'):
            img1 = img1[2:]
        if img2.startswith('./'):
            img2 = img2[2:]
        
        img1_path = f'./selected_samples/{img1}'
        img2_path = f'./selected_samples/{img2}'
        
        try:
            similarity = float(sim_str)
        except ValueError:
            print(f"Warning: Invalid similarity value '{sim_str}'. Set to 0.")
            similarity = 0.0
        
        pairs.append((img1_path, img2_path, similarity))
    return pairs

# ==================== Dataset Class ====================

class SimilarityPairDataset(Dataset):
    def __init__(self, pairs, patch_size=14, device='cpu'):
        self.pairs = pairs
        self.patch_size = patch_size
        self.device = device
        self.cache = {}

    def _compute_feat(self, img_path):
        if img_path in self.cache:
            return self.cache[img_path]
        if not os.path.exists(img_path):
            t = torch.zeros((1, 384), dtype=torch.float32, device=self.device)
            self.cache[img_path] = t
            return t
        feats_list = extract_features([img_path], return_patches=True)
        if not feats_list:
            t = torch.zeros((1, 384), dtype=torch.float32, device=self.device)
            self.cache[img_path] = t
            return t
        first = feats_list[0]
        if isinstance(first, tuple):
            feat_np = first[0]
        else:
            feat_np = first
        if feat_np is None or getattr(feat_np, 'size', 1) == 0:
            t = torch.zeros((1, 384), dtype=torch.float32, device=self.device)
            self.cache[img_path] = t
            return t
        t = torch.from_numpy(np.array(feat_np, dtype=np.float32)).float().to(self.device)
        self.cache[img_path] = t
        return t

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2, sim = self.pairs[idx]
        feat1 = self._compute_feat(img1)
        feat2 = self._compute_feat(img2)
        return feat1, feat2, torch.tensor(float(sim), dtype=torch.float32, device=self.device)

def collate_fn(batch):
    feat1_list, feat2_list, sim_list = [], [], []
    for feat1, feat2, sim in batch:
        feat1_list.append(feat1)
        feat2_list.append(feat2)
        sim_list.append(sim.unsqueeze(0))
    sim_tensor = torch.cat(sim_list, dim=0)
    return feat1_list, feat2_list, sim_tensor

# ==================== Model Definition ====================

class GraphBasedSimilarityModel(nn.Module):
    def __init__(self, feature_dim=384, graph_hidden_dim=256, style_dim=64):
        super().__init__()
        self.graph_encoder = nn.ModuleDict({
            'node_proj': nn.Linear(feature_dim, graph_hidden_dim),
            'graph_conv1': nn.Linear(graph_hidden_dim, graph_hidden_dim),
            'graph_conv2': nn.Linear(graph_hidden_dim, graph_hidden_dim),
            'pooling': nn.AdaptiveAvgPool1d(1)
        })
        self.style_encoder = nn.Sequential(
            nn.Linear(graph_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, style_dim),
            nn.Tanh()
        )
        self.similarity_head = nn.Sequential(
            nn.Linear(style_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def encode_image(self, patch_features):
        node_features = self.graph_encoder['node_proj'](patch_features)
        node_features = F.relu(self.graph_encoder['graph_conv1'](node_features))
        node_features = F.relu(self.graph_encoder['graph_conv2'](node_features))
        graph_representation = self.graph_encoder['pooling'](
            node_features.transpose(0, 1)
        ).squeeze(-1)
        style_embedding = self.style_encoder(graph_representation)
        return style_embedding

    def forward(self, img1_patches, img2_patches):
        style1 = self.encode_image(img1_patches)
        style2 = self.encode_image(img2_patches)
        combined = torch.cat([style1, style2], dim=-1)
        similarity = self.similarity_head(combined)
        return similarity.squeeze()

# ==================== Training Functions ====================

def train_model(model, pairs, device, val_pairs=None, epochs=5, lr=1e-4, 
                weight_decay=1e-5, log_interval=20, batch_size=32, 
                restore_best_weights=True):
    model.to(device)
    model.train()
    dataset = SimilarityPairDataset(pairs, patch_size=patch_size, device=device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for batch_idx, (feat1_list, feat2_list, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for i in range(len(feat1_list)):
                pred = model(feat1_list[i], feat2_list[i])
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                batch_loss += criterion(pred, targets[i].unsqueeze(0))
            
            avg_batch_loss = batch_loss / len(feat1_list)
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_loss += avg_batch_loss.item()
            
            if (batch_idx + 1) % log_interval == 0:
                print(f"Epoch[{epoch}] batch[{batch_idx+1}/{len(dataloader)}] loss={avg_batch_loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        t1 = time.time()
        print(f"Epoch {epoch} finished, avg_loss={avg_epoch_loss:.6f}, time={(t1-t0):.1f}s")
        
        val_loss = None
        if val_pairs is not None:
            val_loss, _ = test_model(model, val_pairs, device, batch_size=batch_size)
            print(f"Epoch {epoch} validation loss={val_loss:.6f}")
        
        if val_loss is not None:
            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, MODEL_CHECKPOINT_PATH)
                print(f"Saved best model (val_loss {best_val_loss:.6f})")
        else:
            if avg_epoch_loss < best_val_loss:
                best_val_loss = avg_epoch_loss
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, MODEL_CHECKPOINT_PATH)
                print(f"Saved best model (train_loss {best_val_loss:.6f})")
    
    if restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)
        print("Restored model to best weights")
    
    return model

def test_model(model, test_pairs, device, batch_size=32):
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    records = []
    
    test_dataset = SimilarityPairDataset(test_pairs, patch_size=14, device=device)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    with torch.no_grad():
        for feat1_list, feat2_list, targets in test_loader:
            for i in range(len(feat1_list)):
                pred = model(feat1_list[i], feat2_list[i])
                if pred.dim() == 0:
                    pred_scalar = pred.item()
                    pred_tensor = pred.unsqueeze(0)
                else:
                    pred_scalar = float(pred.cpu().numpy().ravel()[0])
                    pred_tensor = pred
                
                loss = criterion(pred_tensor, targets[i].unsqueeze(0))
                losses.append(loss.item())
                records.append((
                    test_pairs[i][0],
                    test_pairs[i][1],
                    float(targets[i].item()),
                    float(pred_scalar)
                ))
    
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, records

# ==================== Main Execution ====================

if __name__ == "__main__":
    print(f"Using device: {device}")
    
    # Load data
    pairs = load_similarity_pairs(txt_path='./similarity_all.txt')
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    print(f"Total pairs={len(pairs)}, train={len(train_pairs)}, test={len(test_pairs)}")
    
    # Initialize model
    model = GraphBasedSimilarityModel(feature_dim=384, graph_hidden_dim=256, style_dim=64)
    
    # Check if checkpoint exists
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Loading saved model from {MODEL_CHECKPOINT_PATH}")
        try:
            checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint)
            print("Model loaded successfully")
            
            # Test loaded model
            avg_loss, records = test_model(model, test_pairs, device=device, batch_size=32)
            print(f"Loaded model test loss: {avg_loss:.6f}")
            
            # Print some predictions
            for rec in records[:5]:
                img1, img2, true, pred = rec
                print(f"{os.path.basename(img1)} <-> {os.path.basename(img2)} | true={true:.4f} pred={pred:.4f}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training from scratch...")
            model = train_model(
                model, train_pairs, device=device, 
                val_pairs=test_pairs, epochs=25, 
                lr=1e-4, batch_size=32, 
                restore_best_weights=True
            )
    else:
        print(f"No checkpoint found at {MODEL_CHECKPOINT_PATH}. Training from scratch...")
        model = train_model(
            model, train_pairs, device=device, 
            val_pairs=test_pairs, epochs=25, 
            lr=1e-4, batch_size=32, 
            restore_best_weights=True
        )
    
    # Final test
    avg_loss, records = test_model(model, test_pairs, device=device, batch_size=32)
    print(f"Final test avg_loss={avg_loss:.6f}")
    
    # Print sample results
    print("\nSample predictions:")
    for rec in records[:10]:
        img1, img2, true, pred = rec
        img1_name = os.path.basename(img1)
        img2_name = os.path.basename(img2)
        print(f"{img1_name} <-> {img2_name} | true={true:.4f} pred={pred:.4f}")