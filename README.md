# similarity_analysis
a promising tool to distinguish the style similarity of two images created by Xu Kaiyun

# Create Python 3.9.25 environment named 'dinov2'
conda create -n dinov2 python=3.9.25 -y
conda activate dinov2

# Install system libraries and CUDA toolkits via conda
conda install -c conda-forge -c pytorch -c nvidia \
    pytorch=2.0.0 torchvision=0.15.0 cudatoolkit=11.7 \
    ffmpeg opencv matplotlib-base scikit-learn \
    numpy=1.26.4 pandas=2.2.2 scipy=1.13.1 \
    pillow tesseract openjpeg libtiff \
    tbb mkl mkl-include mkl-service \
    -y
    
# Ensure you are in the dinov2 environment
conda activate dinov2

# Upgrade pip first
pip install --upgrade pip

# Install all packages (this may take 20-40 minutes)
pip install -r requirements.txt

# If installation fails due to dependency conflicts, try:
pip install -r requirements.txt --no-deps


# If you want to train the GNN-based method
python train2.py
# if you want to change the dataset
#Step1 Your similarity dataset should be stored in a text file  
#Format 1: Tab-separated
image1.jpg    image2.jpg    0.85
image3.jpg    image4.jpg    0.42

#Format 2: Comma-separated
image1.jpg,image2.jpg,0.85
image3.jpg,image4.jpg,0.42

#Format 3: Mixed (the code handles both)
image1.jpg  image2.jpg,0.85
image3.jpg,image4.jpg    0.42
#Step2 prepare your data(jpg)
your_project/
├── similarity_all.txt          # Your similarity pairs file
├── selected_samples/           # Image directory
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ...
├── train_similarity_network.py # The training script
└── best_neural_network_model.pt # Saved model (will be created)

#Step3 :Change this line in train2.py
pairs = load_similarity_pairs(txt_path='./similarity_all.txt')

# To use your file:
pairs = load_similarity_pairs(txt_path='./your_similarity_file.txt')

# If you want to train the Neural-Network-based method  
python train1.py
# if you want to change the dataset follow the three step shown above



# If you want to inference the GNN-based method 
python inference2.py --input test_style.txt  --model best_graph_similarity_model_b.pt --output predictions_test_style.csv --base_dir ./selected_samples

# Full command with all options
python graph_similarity_inference.py \
    --input similarity_pairs.txt \
    --model best_graph_similarity_model.pt \
    --base_dir ./my_images \
    --output predictions.csv \
    --device cuda
    
    

#Step1 Your similarity dataset should be stored in a text file 
# Format: image1_path image2_path [similarity_score]
# With similarity scores (for evaluation)
image1.jpg image2.jpg 0.85
cat.png dog.jpg 0.42
landscape/photo1.jpeg portrait/photo2.png 0.73

# Without similarity scores (for prediction only)
building_a.jpg building_b.jpg
style1.png style2.png

#Step2 prepare your data(jpg)
your_project/
├── graph_similarity_inference.py      # Main script
├── best_graph_similarity_model.pt     # Pre-trained model
├── my_images/                         # Your image directory
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── cat.png
│   ├── dog.jpg
│   ├── landscape/
│   │   └── photo1.jpeg
│   └── portrait/
│       └── photo2.png
└── similarity_pairs.txt              # Your dataset file
