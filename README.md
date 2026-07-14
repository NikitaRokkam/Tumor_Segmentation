# Tumor Segmentation with Attention U-Net

A deep learning pipeline for breast ultrasound tumor segmentation using an Attention U-Net with a pretrained EfficientNet-B4 encoder. The project covers the complete workflow from data preparation and model training to evaluation, model export, and API-based inference.

## Overview

This project uses the BUSI (Breast Ultrasound Images) dataset to perform binary semantic segmentation, identifying tumor regions in ultrasound images. The model combines the localization strengths of U-Net with attention-gated skip connections and transfer learning from EfficientNet-B4.

### Key Features

* Attention U-Net architecture
* Pretrained EfficientNet-B4 encoder
* Dice + Binary Cross-Entropy loss
* PyTorch training and evaluation pipeline
* TorchScript model export
* FastAPI inference service
* Dockerized deployment
* Automated tests for core components

---

## Dataset

The project uses the BUSI (Breast Ultrasound Images) dataset, which contains benign, malignant, and normal breast ultrasound scans with corresponding segmentation masks.

For segmentation, all lesion masks are treated as a single foreground class, resulting in a binary segmentation task:

* Background
* Tumor

Dataset split:

* Training: 70%
* Validation: 15%
* Test: 15%

A fixed random seed is used to ensure reproducibility.

---

## Model Architecture

The segmentation model consists of:

* **Encoder:** EfficientNet-B4 pretrained on ImageNet
* **Decoder:** U-Net style upsampling path
* **Attention Gates:** Applied to skip connections to emphasize relevant spatial features
* **Output Head:** Single-channel segmentation mask logits

Using a pretrained encoder improves feature extraction and convergence, particularly when training on relatively small medical imaging datasets.

---

## Training

Training is implemented using a standard PyTorch workflow.

### Loss Function

A hybrid loss combines Dice Loss and Binary Cross-Entropy:

```python
loss = 0.5 * dice_loss + 0.5 * bce_loss
```

This balances overlap optimization with stable gradient updates during training.

### Metrics

Model performance is evaluated using:

* Dice Coefficient
* Intersection over Union (IoU)
* Pixel Accuracy

Dice coefficient is used as the primary segmentation metric.

---

## Evaluation

The evaluation pipeline:

* Runs inference on the held-out test set
* Reports Dice, IoU, and Pixel Accuracy
* Saves prediction visualizations for qualitative analysis
* Identifies difficult examples for further inspection

---

## Deployment

The trained model can be exported to TorchScript for inference.

### FastAPI Endpoints

**POST /v1/predict**

* Accepts ultrasound images
* Returns predicted segmentation mask
* Reports tumor area statistics
* Returns inference latency

**GET /health**

* Health and readiness check

---

## Security Considerations

The API includes basic safeguards suitable for a technical demonstration:

* API key authentication
* File type validation
* File size limits
* No image persistence
* Minimal logging without patient-identifiable information

---

## Project Structure

```text
src/
├── config.py
├── dataset.py
├── model.py
├── losses.py
├── metrics.py
├── train.py
├── evaluate.py
└── export_model.py

api/
└── main.py

tests/
└── test_core.py

Dockerfile
docker-compose.yml
requirements.txt
```

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train

```bash
python -m src.train
```

### Evaluate

```bash
python -m src.evaluate
```

### Export TorchScript Model

```bash
python -m src.export_model
```

### Run Tests

```bash
pytest tests/
```

### Start API

```bash
uvicorn api.main:app --reload
```

---

## Future Improvements

* K-fold cross-validation
* Hyperparameter optimization
* Stronger augmentation strategies
* Multi-task learning for segmentation and classification
* DICOM support and automated de-identification
* Production monitoring and drift detection

---

## Technologies

* Python
* PyTorch
* TorchVision
* FastAPI
* Docker
* TorchScript
* NumPy
* Matplotlib
