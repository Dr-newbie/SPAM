# SPAM : Spatial transcriptomics Predictor with self-supervised Alignment of Multimodalities

> **Integrating gene expression, spatial coordinates and histological features  
> for predicting spatially resolved genes via Self-Supervised Learning**
>
> Jaeyun Park, Dongsin Kim, Minsik Oh*  
> College of Data Technology, Myongji University

---

## 1. Overview 🧬

**SPAM** is a multimodal framework for **predicting spatially resolved gene expression** from:

- Histology image features (H&E patches, foundation models)
- Spatial coordinates (cell / spot locations)
- Gene expression profiles

The training pipeline consists of:

1. **Contrastive pretraining**  
   - Jointly learns representations of image, coordinates, and gene expression  
   - Uses a foundation image encoder + GCN + gene encoder

2. **Cross-attention + ZINB finetuning**  
   - Applies cross-attention between modalities (image ↔ coord, image ↔ gene)  
   - Merges attended features and reconstructs gene expression with a **ZINB decoder**

3. **Inference**  
   - Uses the finetuned model to predict gene expression for new sections  
   - Saves results (predicted expression, evaluation, plots, etc.)

---

## 2. Repository Structure 📁

(High-level description; file names may be updated.)

```text
SPAM/
├── models/          # Core model components
│   ├── Foundations.py      # Image foundation encoder wrapper (UNI, H-optimus, etc.)
│   ├── gene_encoder.py     # Gene expression encoder
│   ├── GCN_update.py       # Spatial GCN for coordinates
│   ├── contrastive.py      # Contrastive pretraining modules
│   └── ...
├── utils/           # Utilities
│   ├── dataset.py          # Dataset & dataloader
│   ├── graph_construction.py  # KNN graph building
│   ├── lora_utils.py       # (Optional) LoRA utilities
│   ├── loss_util.py        # ZINB loss, contrastive loss, etc.
│   └── ...
├── alignment.ipynb  # Example / debugging notebook for alignment
├── main.py          # Entry point for pretraining & finetuning
├── inference.py     # Entry point for inference (prediction)
└── README.md
