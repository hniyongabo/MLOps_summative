---
title: Bean Leaf Doctor
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 🌱 Bean Leaf Doctor: bean leaf disease detection, end-to-end ML pipeline

Beans are a key source of protein in East Africa, and two leaf diseases, **angular leaf spot** and **bean rust**, can destroy much of a harvest. This project classifies a **photo of a bean leaf** into one of 3 classes: `angular_leaf_spot`, `bean_rust` or `healthy`. The model is served in a web app where you can:

- **Predict:** upload one leaf photo and get the diagnosis with probabilities
- **Retrain:** upload new labelled photos, which are **saved to a database**, **preprocessed**, and used to **fine-tune the existing (pre-trained) model**
- **Explore:** a dashboard with model metrics and **data insights / visualisations**

| | |
|---|---|
| 🎥 **Video demo** | _paste your YouTube link here_ |
| 🌐 **Live app** | _paste your Hugging Face Space URL here_ |
| 📓 **Notebook** | [`notebook/bean_disease_classifier.ipynb`](notebook/bean_disease_classifier.ipynb) |

---

## Dataset
**iBean** by the Makerere University AI Lab and the National Crops Resources Research Institute (NaCRRI), Uganda: 1,295 field photos taken with smartphones and labelled by experts. Splits used in this project (images resized to 224×224):

| Folder | Purpose | Images |
|---|---|---|
| `data/train/<class>/` | 75% of the official train split; builds the model in the notebook | ≈775 |
| `data/retrain_pool/<class>/` | 25% of the official train split; *new data* for the retraining demo (zipped as `demo_samples/new_data_for_retraining.zip`) | ≈259 |
| `data/validation/<class>/` | early stopping and hyper-parameter tuning | 133 |
| `data/test/<class>/` | final evaluation only | 128 |
| `demo_samples/` | 3 test photos per class for the prediction demo; the true class is in the file name | 9 |

## Model
**Baseline:** a vanilla CNN trained from scratch, with SGD and no regularisation, augmentation or tuning.

**Optimised model:** transfer learning with **MobileNetV2 pre-trained on ImageNet**, using these optimisation techniques:

| Technique | Implementation |
|---|---|
| Pre-trained model | MobileNetV2 feature extractor, frozen first, then the top 40 layers fine-tuned |
| Data augmentation | random flip / rotation / zoom / contrast |
| Regularization | Dropout + L2 weight penalty on the classifier |
| Optimizer | Adam |
| Early stopping | monitor validation loss, restore best weights |
| Learning-rate scheduling | ReduceLROnPlateau |
| Hyper-parameter tuning | grid search over learning rate × dropout |

**Evaluation metrics:** Accuracy, Loss, Precision, Recall, F1 score, ROC-AUC, plus the confusion matrix, classification report and per-class metrics.

| Test set (128 images) | Accuracy | Loss | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Vanilla CNN | _fill in from notebook section 7_ | | | | |
| MobileNetV2 v1 | _fill in_ | | | | |
| After retraining (v2) | _fill in from the Retrain page_ | | | | |

## Project structure
```
├── app.py                         # Flask web app: dashboard, /predict, /retrain + JSON API
├── notebook/bean_disease_classifier.ipynb   # data, preprocessing, training, tuning, evaluation (run in Colab)
├── src/
│   ├── preprocessing.py           # image preprocessing (shared by notebook, prediction, retraining)
│   ├── model.py                   # metrics, save/load, fine-tuning
│   ├── prediction.py              # predict one image
│   ├── retrain.py                 # upload -> database -> preprocess -> retrain pipeline
│   ├── database.py                # SQLite: uploaded images + retraining history
│   └── insights.py                # dashboard charts
├── models/                        # bean_model.keras (pre-trained model for retraining), metadata, history
├── data/                          # train / retrain_pool / validation / test images
├── demo_samples/                  # demo photos + retraining zip
├── templates/, static/            # web UI
├── scripts/prepare_data.py        # download + organise the dataset locally (optional)
├── scripts/deploy_to_hf.py        # deploy to Hugging Face Spaces
└── Dockerfile, requirements.txt
```

## 1. Train the model (Google Colab)
1. Open `notebook/bean_disease_classifier.ipynb` in [Google Colab](https://colab.research.google.com) (*File → Upload notebook*).
2. *Runtime → Change runtime type → **T4 GPU***, then *Runtime → Run all* (about 15–25 min).
3. The last cell downloads **`bean_project_outputs.zip`**. Unzip it into the root of this repository; it adds `data/`, `models/` and `demo_samples/`.
4. *File → Download → Download .ipynb* and replace `notebook/bean_disease_classifier.ipynb` with this version, which includes the outputs.

## 2. Run the app locally
Requires Python 3.10–3.12.
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                     # open http://localhost:7860
```
If the model fails to load because of a version mismatch, pin TensorFlow in `requirements.txt` to the version printed in the notebook's first cell (e.g. `tensorflow==2.19.0`).

## 3. Run with Docker
```bash
docker build -t bean-leaf-doctor .
docker run -p 7860:7860 bean-leaf-doctor      # open http://localhost:7860
```

## 4. Deploy for a public URL (Hugging Face Spaces, free)
```bash
pip install huggingface_hub
python scripts/deploy_to_hf.py <your-hf-username>/bean-leaf-doctor
```
You'll need a free Hugging Face account and an access token with *write* permission. The Space builds the Dockerfile; after about 10 minutes the app is live at `https://<your-hf-username>-bean-leaf-doctor.hf.space`.

## How to use the app
**Predict** (`/predict`): upload a leaf photo, e.g. `demo_samples/true_bean_rust_1.jpg`, and click **Predict**. You get the class, the confidence, the probability of each class and what the result means.

**Retrain** (`/retrain`):
1. **Upload & save to database:** upload `demo_samples/new_data_for_retraining.zip`, which has folders `angular_leaf_spot/`, `bean_rust/` and `healthy/` with images inside. Every image is stored in SQLite (`database/mlop.db`, table `training_images`).
2. **Preprocess:** the stored images go through the same preprocessing as for prediction: EXIF rotation, RGB conversion, resizing to 224×224 and array conversion. The pixel scaling to −1…1 happens inside the model.
3. **Retrain:** the saved model `models/bean_model.keras` is loaded as the **pre-trained model** and fine-tuned on the new images plus a sample of the original images, using Adam and early stopping. Training runs in the background and the page shows live progress per epoch. The new version replaces the live model only if it is at least as accurate on the test set; the old version is archived in `models/archive/`.

The dashboard then shows the new metrics, updated charts and the retraining history.

## API (also usable from Postman)
| Method | Endpoint | Body | Description |
|---|---|---|---|
| GET | `/api/status` | – | uptime, model version and metrics, database stats |
| POST | `/api/predict` | form-data `file` = image | predicted class + probabilities |
| POST | `/api/upload` | form-data `file` = .zip | save labelled images to the database |
| POST | `/api/preprocess` | – | preprocess the new images in the database |
| POST | `/api/retrain` | – | start fine-tuning in the background |
| GET | `/api/retrain/status` | – | progress, log and before/after metrics |

## Credits
Dataset: Makerere AI Lab, *Bean disease dataset* (2020), https://github.com/AI-Lab-Makerere/ibean/
