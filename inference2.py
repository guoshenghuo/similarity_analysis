import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as pth_transforms
import hubconf
import numpy as np
import pandas as pd
import os
import argparse
import time
import sys

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

patch_size = 14

def load_dinov2():
    print("Loading DINOv2 model...")
    dinov2_vits14 = hubconf.dinov2_vits14()
    dinov2_vits14 = dinov2_vits14.to(device)
    dinov2_vits14.eval()
    return dinov2_vits14

def adjust_image_size(img, patch_size=14):
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
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    
    arr_filled = arr.copy()
    
    for dim in range(arr_filled.ndim):
        mask = np.isnan(arr_filled)
        if not np.any(mask):
            break
        
        if arr_filled.ndim == 2:
            for i in range(arr_filled.shape[dim]):
                if dim == 0:
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
                    col = arr_filled[:, i]
                    nan_mask = np.isnan(col)
                    if np.any(nan_mask):
                        valid_idx = np.where(~nan_mask)[0]
                        if len(valid_idx) == 0:
                            arr_filled[:, i] = 0.0
                        else:
                            interp_vals = np.interp(
                                np.where(nan_mask)[0],
                                valid_idx,
                                col[valid_idx]
                            )
                            arr_filled[nan_mask, i] = interp_vals
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
    
    arr_filled = np.nan_to_num(arr_filled, nan=0.0, posinf=1e3, neginf=-1e3)
    return arr_filled

def extract_features_single(img_path, dinov2_model):
    try:
        pil = Image.open(img_path).convert("RGB")
        pil = adjust_image_size(pil, patch_size=patch_size)
        
        w, h = pil.size
        h_grid = h // patch_size
        w_grid = w // patch_size
        
        transform = pth_transforms.Compose([
            pth_transforms.Resize(pil.size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        img_tensor = transform(pil)[:3].unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = dinov2_model(img_tensor, return_patches=True)[0]
        
        features_np = features.detach().cpu().numpy()
        
        if np.isnan(features_np).any() or np.isinf(features_np).any():
            features_np = fill_nan_with_interpolation(features_np)
        
        return features_np, (h_grid, w_grid)
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return np.zeros((1, 384)), (1, 1)

def build_spatial_grid_graph(feature_tensor, grid_size, device='cpu'):
    h, w = grid_size
    num_nodes = h * w
    
    if feature_tensor.shape[0] != num_nodes:
        if feature_tensor.shape[0] > num_nodes:
            feature_tensor = feature_tensor[:num_nodes, :]
        else:
            padding = np.zeros((num_nodes - feature_tensor.shape[0], feature_tensor.shape[1]))
            feature_tensor = np.vstack([feature_tensor, padding])
    
    feature_tensor = torch.FloatTensor(feature_tensor).to(device)
    
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
        pos=pos,
        batch=torch.zeros(num_nodes, dtype=torch.long, device=device)
    )
    
    return data

class GraphBasedSimilarityModel(nn.Module):
    def __init__(self, feature_dim=384, hidden_dim=256, style_dim=64):
        super().__init__()
        
        self.pos_embedding = nn.Linear(2, feature_dim)
        self.conv1 = GCNConv(feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.style_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, style_dim),
            nn.Tanh()
        )
        
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
    
    def forward(self, data1, data2):
        x1, pos1, edge_index1, batch1 = data1.x, data1.pos, data1.edge_index, data1.batch
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

class GraphSimilarityInference:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Initializing inference on device: {self.device}")
        
        self.model = self._load_model(model_path)
        self.dinov2_model = load_dinov2()
    
    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = GraphBasedSimilarityModel(
            feature_dim=384, 
            hidden_dim=256,
            style_dim=64
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded from: {model_path}")
        return model
    
    def preprocess_image(self, img_path):
        features, grid_size = extract_features_single(
            img_path, 
            self.dinov2_model
        )
        
        graph_data = build_spatial_grid_graph(features, grid_size, self.device)
        
        return graph_data
    
    def predict_pair(self, img1_path, img2_path):
        if not os.path.exists(img1_path):
            print(f"Warning: Image file not found: {img1_path}")
            return None
        
        if not os.path.exists(img2_path):
            print(f"Warning: Image file not found: {img2_path}")
            return None
        
        print(f"Processing: {os.path.basename(img1_path)}")
        graph1 = self.preprocess_image(img1_path)
        
        print(f"Processing: {os.path.basename(img2_path)}")
        graph2 = self.preprocess_image(img2_path)
        
        with torch.no_grad():
            similarity = self.model(graph1, graph2)
            
            if similarity.dim() == 0:
                similarity_score = similarity.item()
            else:
                similarity_score = float(similarity.cpu().numpy().ravel()[0])
        
        return similarity_score
    
    def batch_predict(self, pairs_list, output_file=None):
        results = []
        
        for i, (img1_path, img2_path, true_similarity) in enumerate(pairs_list):
            print(f"Processing pair {i+1}/{len(pairs_list)}: {os.path.basename(img1_path)} <-> {os.path.basename(img2_path)}")
            
            pred_similarity = self.predict_pair(img1_path, img2_path)
            
            if pred_similarity is not None:
                result = {
                    'img1': img1_path,
                    'img2': img2_path,
                    'true_similarity': true_similarity,
                    'pred_similarity': pred_similarity
                }
                
                if true_similarity is not None:
                    result['error'] = abs(true_similarity - pred_similarity)
                    print(f"  True: {true_similarity:.4f}, Pred: {pred_similarity:.4f}, Error: {abs(true_similarity - pred_similarity):.4f}")
                else:
                    print(f"  Pred: {pred_similarity:.4f}")
                
                results.append(result)
        
        if output_file and results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            if 'true_similarity' in df_results.columns and not df_results['true_similarity'].isna().all():
                mae = df_results['error'].mean()
                print(f"Mean Absolute Error (MAE): {mae:.4f}")
        
        return results
import re
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

    
def main():
    parser = argparse.ArgumentParser(description='Graph Neural Network Image Similarity Inference')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input file path (format: img1 img2 [similarity])')
    parser.add_argument('--model', type=str, default='best_graph_similarity_model.pt',
                       help='Model weight file path')
    parser.add_argument('--base_dir', type=str, default='./selected_samples',
                       help='Image base directory')
    parser.add_argument('--output', type=str, default='graph_similarity_predictions.csv',
                       help='Output results file path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Warning: Model file {args.model} not found")
        print("Please specify a valid model path with --model argument")
        return
    
    print("=" * 50)
    print("Graph Similarity Inference")
    print("=" * 50)
    
    try:
        inference = GraphSimilarityInference(
            model_path=args.model,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing inference: {e}")
        return
    
    try:
        pairs = load_similarity_pairs(args.input)
        print(len(pairs))
    except Exception as e:
        print(f"Error loading pairs: {e}")
        return
    
    print("\nStarting inference...")
    start_time = time.time()
    
    results = inference.batch_predict(pairs, args.output)
    
    end_time = time.time()
    print(f"\nInference completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per pair: {(end_time - start_time) / len(pairs):.2f} seconds")

if __name__ == "__main__":
    main()