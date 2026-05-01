import torch
import os
import sys
import time
import copy
import warnings
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
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

# Add FastSAM path
sys.path.append("/home/kaiyunxu/dinov2/FastSAM")
from fastsam import FastSAM

# ==================== Configuration ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "best_graph_similarity_model.pt"
PATCH_SIZE = 14
N_COMPONENTS = 32
FEATURE_DIM = 384

# ==================== Model Loading ====================
dinov2_vits14 = hubconf.dinov2_vits14().to(DEVICE)

# Load FastSAM model
fastsam_model = FastSAM("/home/kaiyunxu/dinov2/FastSAM/FastSAM_X.pt")

# ==================== Utility Functions ====================

def adjust_image_size(img, patch_size=14):
    w, h = img.size
    new_w = w - (w % patch_size)
    new_h = h - (h % patch_size)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    return img.crop((left, top, right, bottom))

def fill_nan_with_interpolation(arr):
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    
    arr_filled = arr.copy()
    
    if arr_filled.ndim == 2:
        for i in range(arr_filled.shape[0]):
            row = arr_filled[i, :]
            nan_mask = np.isnan(row)
            if np.any(nan_mask):
                valid_idx = np.where(~nan_mask)[0]
                if len(valid_idx) == 0:
                    arr_filled[i, :] = 0.0
                else:
                    interp_vals = np.interp(
                        np.where(nan_mask)[0],
                        valid_idx,
                        row[valid_idx]
                    )
                    arr_filled[i, nan_mask] = interp_vals
    else:
        flat_arr = arr_filled.flatten()
        nan_mask = np.isnan(flat_arr)
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) > 0:
            interp_vals = np.interp(
                np.where(nan_mask)[0],
                valid_idx,
                flat_arr[valid_idx]
            )
            flat_arr[nan_mask] = interp_vals
            arr_filled = flat_arr.reshape(arr_filled.shape)
        else:
            arr_filled = np.zeros_like(arr_filled)
    
    return np.nan_to_num(arr_filled, nan=0.0, posinf=1e3, neginf=-1e3)

def extract_features(image_paths, return_patches=False):
    features_list = []
    grid_sizes = []
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            img = adjust_image_size(img, PATCH_SIZE)
            w, h = img.size
            h_grid = h // 14
            w_grid = w // 14
            grid_sizes.append((h_grid, w_grid))

            transform = pth_transforms.Compose([
                pth_transforms.Resize(img.size),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            img_tensor = transform(img)[:3].unsqueeze(0).to(DEVICE)
            
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
            default_feat = np.zeros((1, FEATURE_DIM))
            features_list.append(default_feat)
            grid_sizes.append((1, 1))
    
    return features_list, grid_sizes

def build_spatial_grid_graph(feature_tensor, grid_size, device='cpu'):
    h, w = grid_size
    num_nodes = h * w
    
    if feature_tensor.shape[0] != num_nodes:
        if feature_tensor.shape[0] > num_nodes:
            feature_tensor = feature_tensor[:num_nodes, :]
        else:
            padding = torch.zeros((num_nodes - feature_tensor.shape[0], feature_tensor.shape[1]), 
                                device=device)
            feature_tensor = torch.cat([feature_tensor, padding], dim=0)
    
    node_indices = torch.arange(num_nodes, device=device).view(h, w)
    
    edges_src = []
    edges_dst = []
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for dy, dx in neighbors:
        src_r_min = max(0, -dy)
        src_r_max = min(h, h - dy)
        src_c_min = max(0, -dx)
        src_c_max = min(w, w - dx)
        
        dst_r_min = max(0, dy)
        dst_r_max = min(h, h + dy)
        dst_c_min = max(0, dx)
        dst_c_max = min(w, w + dx)
        
        src_nodes = node_indices[src_r_min:src_r_max, src_c_min:src_c_max].flatten()
        dst_nodes = node_indices[dst_r_min:dst_r_max, dst_c_min:dst_c_max].flatten()
        
        edges_src.append(src_nodes)
        edges_dst.append(dst_nodes)

    edge_index = torch.stack([
        torch.cat(edges_src),
        torch.cat(edges_dst)
    ], dim=0)

    y_coords = torch.linspace(0, 1, h, device=device).view(h, 1).repeat(1, w).flatten()
    x_coords = torch.linspace(0, 1, w, device=device).view(1, w).repeat(h, 1).flatten()
    pos = torch.stack([x_coords, y_coords], dim=1)

    data = Data(
        x=feature_tensor,
        edge_index=edge_index,
        pos=pos
    )
    
    return data

def load_similarity_pairs(txt_path='./similarity.txt'):
    pairs = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for line_num, line in enumerate(lines, 1):
        parts = re.split(r'[\t,]', line)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) != 3:
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
            similarity = 0.0
        
        pairs.append((img1_path, img2_path, similarity))
    return pairs

# ==================== Dataset Class ====================

class SimilarityPairDataset(Dataset):
    def __init__(self, pairs, device='cpu'):
        self.pairs = pairs
        self.device = device
        self.cache = {}
    
    def __len__(self):
        return len(self.pairs)
    
    def _compute_feat(self, img_path):
        if img_path in self.cache:
            return self.cache[img_path]
        
        try:
            feats_list, grids = extract_features([img_path], return_patches=True)
            
            if (feats_list is None or len(feats_list) == 0 or 
                feats_list[0] is None or feats_list[0].size == 0):
                feat_np = np.zeros((1, FEATURE_DIM), dtype=np.float32)
                grid = (1, 1)
            else:
                feat_np = feats_list[0]
                grid = grids[0] if grids and len(grids) > 0 else (1, 1)
            
            feat_tensor = torch.tensor(feat_np, dtype=torch.float32).to(self.device)
            result = (feat_tensor, grid)
            self.cache[img_path] = result
            return result
        
        except Exception as e:
            print(f"Warning: Failed to process {img_path} - {str(e)}")
            feat_tensor = torch.zeros((1, FEATURE_DIM), dtype=torch.float32).to(self.device)
            result = (feat_tensor, (1, 1))
            self.cache[img_path] = result
            return result
    
    def __getitem__(self, idx):
        img1_path, img2_path, sim = self.pairs[idx]
        
        feat1, grid1 = self._compute_feat(img1_path)
        feat2, grid2 = self._compute_feat(img2_path)
        
        data1 = build_spatial_grid_graph(feat1, grid1, device=self.device)
        data2 = build_spatial_grid_graph(feat2, grid2, device=self.device)
        
        label = torch.tensor(float(sim), dtype=torch.float32).to(self.device)
        
        return data1, data2, label

def collate_fn(batch):
    data1_list = [item[0] for item in batch]
    data2_list = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    batch1 = Batch.from_data_list(data1_list)
    batch2 = Batch.from_data_list(data2_list)
    labels = torch.stack(labels)
    
    return batch1, batch2, labels

# ==================== Model Definition ====================

class GraphBasedSimilarityModel(nn.Module):
    def __init__(self, feature_dim=384, hidden_dim=128, style_dim=64):
        super().__init__()
        
        self.pos_embedding = nn.Linear(2, feature_dim)
        self.conv1 = GCNConv(feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.style_head = nn.Sequential(
            nn.Linear(hidden_dim, style_dim),
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
    
    def _check_nan_inf(self, tensor, tensor_name, step_desc):
        if tensor is None:
            return
        
        tensor_cpu = tensor.detach().cpu()
        has_nan = torch.isnan(tensor_cpu).any().item()
        has_inf = torch.isinf(tensor_cpu).any().item()
        
        if has_nan or has_inf:
            print(f"Exception at {step_desc}: {tensor_name} has NaN={has_nan}, Inf={has_inf}")
    
    def forward(self, data1, data2):
        x1, pos1, edge_index1, batch1 = data1.x, data1.pos, data1.edge_index, data1.batch
        self._check_nan_inf(x1, "x1", "data1.x")
        
        pos_emb1 = self.pos_embedding(pos1)
        x1 = x1 + pos_emb1
        x1 = F.relu(self.conv1(x1, edge_index1))
        x1 = F.relu(self.conv2(x1, edge_index1))
        graph_emb1 = global_mean_pool(x1, batch1)
        style1 = self.style_head(graph_emb1)
        
        x2, pos2, edge_index2, batch2 = data2.x, data2.pos, data2.edge_index, data2.batch
        pos_emb2 = self.pos_embedding(pos2)
        x2 = x2 + pos_emb2
        x2 = F.relu(self.conv1(x2, edge_index2))
        x2 = F.relu(self.conv2(x2, edge_index2))
        graph_emb2 = global_mean_pool(x2, batch2)
        style2 = self.style_head(graph_emb2)
        
        combined = torch.cat([style1, style2], dim=-1)
        similarity = self.similarity_head(combined)
        
        return similarity.squeeze()

# ==================== Training Functions ====================

def train_model(model, pairs, device, val_pairs=None, epochs=25, lr=1e-4, 
                batch_size=32, log_interval=20):
    model.to(device)
    model.train()
    
    dataset = SimilarityPairDataset(pairs, device=device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        
        for batch_idx, (batch1, batch2, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            pred = model(batch1, batch2)
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)
            
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % log_interval == 0:
                print(f"Epoch[{epoch}] batch[{batch_idx+1}/{len(dataloader)}] loss={loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        t1 = time.time()
        print(f"Epoch {epoch} finished, train_loss={avg_epoch_loss:.6f}, time={(t1-t0):.1f}s")
        
        if val_pairs is not None:
            val_loss, _ = test_model(model, val_pairs, device, batch_size=batch_size)
            print(f"Epoch {epoch} validation loss={val_loss:.6f}")
            
            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, CHECKPOINT_PATH)
                print(f"Saved best model (val_loss {best_val_loss:.6f})")
        else:
            if avg_epoch_loss < best_val_loss:
                best_val_loss = avg_epoch_loss
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, CHECKPOINT_PATH)
                print(f"Saved best model (train_loss {best_val_loss:.6f})")
    
    if best_state is not None:
        model.load_state_dict(best_state)
        print("Restored model to best weights")
    
    return model

def test_model(model, test_pairs, device, batch_size=32):
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    records = []
    
    test_dataset = SimilarityPairDataset(test_pairs, device=device)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    with torch.no_grad():
        for batch_idx, (batch1, batch2, targets) in enumerate(test_loader):
            pred = model(batch1, batch2)
            
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)
            
            loss = criterion(pred, targets)
            losses.append(loss.item())
            
            for i in range(len(pred)):
                idx = batch_idx * batch_size + i
                if idx < len(test_pairs):
                    pred_scalar = pred[i].item()
                    true_scalar = targets[i].item()
                    records.append((
                        test_pairs[idx][0],
                        test_pairs[idx][1],
                        true_scalar,
                        pred_scalar
                    ))
    
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, records

# ==================== Main Execution ====================

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # Load data
    pairs = load_similarity_pairs(txt_path='./similarity_all.txt')
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    print(f"Total pairs={len(pairs)}, train={len(train_pairs)}, test={len(test_pairs)}")
    
    # Initialize model
    model = GraphBasedSimilarityModel(feature_dim=FEATURE_DIM, hidden_dim=256, style_dim=64)
    
    # Check for existing checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading saved model from {CHECKPOINT_PATH}")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint)
            print("Model loaded successfully from checkpoint")
            
            # Test loaded model
            avg_loss, records = test_model(model, test_pairs, device=DEVICE, batch_size=32)
            print(f"Loaded model test loss: {avg_loss:.6f}")
            
            # Print sample predictions
            print("\nSample predictions from loaded model:")
            for i, rec in enumerate(records[:5]):
                img1, img2, true_val, pred_val = rec
                img1_name = os.path.basename(img1)
                img2_name = os.path.basename(img2)
                print(f"{img1_name} <-> {img2_name} | true={true_val:.4f} pred={pred_val:.4f}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training from scratch...")
            model = train_model(
                model, 
                train_pairs, 
                device=DEVICE, 
                val_pairs=test_pairs,
                epochs=25, 
                lr=1e-4,
                batch_size=32
            )
    else:
        print(f"No checkpoint found at {CHECKPOINT_PATH}. Training from scratch...")
        model = train_model(
            model, 
            train_pairs, 
            device=DEVICE, 
            val_pairs=test_pairs,
            epochs=25, 
            lr=1e-4,
            batch_size=32
        )
    
    # Final test
    avg_loss, records = test_model(model, test_pairs, device=DEVICE, batch_size=32)
    print(f"\nFinal test average loss: {avg_loss:.6f}")
    
    # Print sample results
    print("\nSample predictions:")
    for i, rec in enumerate(records[:10]):
        img1, img2, true_val, pred_val = rec
        img1_name = os.path.basename(img1)
        img2_name = os.path.basename(img2)
        print(f"{img1_name} <-> {img2_name} | true={true_val:.4f} pred={pred_val:.4f}")