# FoodVision — Food Image Classifier

A deep learning image classifier that identifies 101 different food categories using transfer learning. The project compares multiple pretrained architectures and includes deep error analysis to understand model behavior.

**[Try the Live Demo](https://huggingface.co/spaces/youresolena/FoodVision)**

---

## Project Overview

This project fine-tunes pretrained CNN models on the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (101 categories, 101,000 images) and compares their performance under identical conditions. The focus is on practical model comparison and in-depth error analysis rather than maximizing a single accuracy number.

## Models Compared

| Model | Approach | Accuracy | Loss |
|---|---|---|---|
| EfficientNet-B2 | Feature Extraction | 54.6% | 1.79 |
| EfficientNet-B0 | Fine-Tuned | 67.3% | 1.22 |
| **ResNet50** | **Fine-Tuned** | **78.0%** | **0.83** |
All models were trained and evaluated on the same data subset under identical conditions for fair comparison.

## Key Findings

- **Fine-tuning significantly outperforms feature extraction.** Unfreezing the last convolutional block improved ResNet50 from 57.0% to 69.9% (+13%).
- **ResNet50 consistently outperformed EfficientNet variants** on this dataset and training setup.
- **Visually similar food categories** (e.g., steak vs pork chop, chocolate cake vs chocolate mousse) account for the majority of misclassifications.
- **Data Normalization is critical.** Models trained with ImageNet normalization values showed significant accuracy improvements over unnormalized inputs.

## Analysis Highlights

- **Confusion Matrix** — Identifies the 20 worst-performing classes and their misclassification patterns
- **Most Confused Pairs** — Top 15 class pairs the model confuses most frequently
- **Most Wrong Predictions** — Images where the model was highly confident but incorrect
- **Best vs Worst Classes** — Per-class accuracy breakdown showing where the model excels and struggles

## Tech Stack

- **PyTorch & torchvision** — Model training and data pipeline
- **EfficientNet-B0, B2 & ResNet50** — Pretrained architectures (ImageNet)
- **Gradio** — Web app deployment
- **Hugging Face Spaces** — Hosting
- **Google Colab** — Training environment (T4 GPU)
- **scikit-learn & seaborn** — Evaluation metrics and visualization

## Project Structure

```
FoodVision/
├── app.py                  # Gradio web app
├── requirements.txt        # Dependencies
├── ResNet50-FineTuned.pth  # Best model weights
├── FoodVision.ipynb        # Full training & analysis notebook
└── README.md
```

## How to Run Locally

```bash
git clone https://huggingface.co/spaces/youresolena/FoodVision
cd FoodVision
pip install -r requirements.txt
python app.py
```

## Future Improvements

- Train on the full Food-101 dataset (101K images) for higher accuracy
- Experiment with learning rate schedulers (CosineAnnealing, ReduceLROnPlateau)
- Add more aggressive data augmentation (ColorJitter, RandomRotation, MixUp)
- Unfreeze more layers progressively during fine-tuning
- Try newer architectures (ConvNeXt, ViT)
- Add TensorBoard experiment tracking

## What I Learned

- Building end-to-end ML pipelines: data loading, preprocessing, training, evaluation, deployment
- Transfer learning: feature extraction vs fine-tuning tradeoffs
- The importance of data normalization and proper data splits
- Systematic model comparison under controlled conditions
- Debugging ML issues: identifying data pipeline problems through accuracy diagnostics
