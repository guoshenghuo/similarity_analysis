# similarity_analysis
A promising tool to distinguish the style similarity of two images.  
Created by Xu Kaiyun.

---
## Demo
We provide a simple demo to show the image style similarity prediction.

### Input Images
<div align="center">
<img src="demos/a1.jpg" width="400">
<img src="demos/a2.jpg" width="400">
</div>

### Prediction Result
- Final Similarity Score: **0.8814**
- Conclusion: **Quite Similar**
---
## Code
Coming soon.
## Environment Setup
This project requires **Python 3.9.25** and a Conda environment.

### 1. Create and activate Conda environment
```bash
conda create -n dinov2 python=3.9.25 -y
conda activate dinov2
```
### 2. Install core dependencies via Conda
```bash
conda install -c conda-forge -c pytorch -c nvidia \
    pytorch=2.0.0 torchvision=0.15.0 cudatoolkit=11.7 \
    ffmpeg opencv matplotlib-base scikit-learn \
    numpy=1.26.4 pandas=2.2.2 scipy=1.13.1 \
    pillow tesseract openjpeg libtiff \
    tbb mkl mkl-include mkl-service \
    -y
```
### 3. Upgrade pip and install Python packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
## Inference
```bash
python inference2.py \
  --input test_style.txt \
  --model best_graph_similarity_model_b.pt \
  --output predictions_test_style.csv \
  --base_dir ./selected_samples
```
