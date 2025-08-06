🧩 Overview
The Diffusion-Guided Completor trains a diffusion model to complete missing modalities.


📁 Project Structure
completor/
├── src/
│   ├── main.py                  # Entry point for pretrain/infer
│   ├── utils/
│   │   └── quick_start.py       # Pretrain & infer logic
│   └── models/
│       └── unified_dif_model.py # Unified model: Unet + diffusion
├── data/
│   └── [dataset]/               
│       ├── image_feat.npy       
│       └── text_feat.npy        
├── checkpoints/                 # Saved Unet weights
└── output/                      # Inferred visual features (.pkl)


🚀 Usage
1. Pretrain Unet on a Dataset:  python src/main.py --mode pretrain --pretrain_dataset sports
2. Infer Features Using Pretrained Unet:  python src/main.py --mode infer --pretrain_dataset sports --infer_dataset sports
