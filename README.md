# SPAC3D - AI Room Intelligence

SPAC3D is a full-stack web application that analyzes room photos using a multi-model machine learning pipeline. It detects furniture objects with bounding boxes, classifies the room into one of six interior design styles, and recommends matching furniture items based on visual similarity.

---

## How It Works

### Overview

A user uploads a room photo through the React frontend. The image is sent to a FastAPI backend where five ML models run in sequence. The results — detected furniture, room style, confidence score, and furniture recommendations — are returned and displayed in the UI.

### ML Pipeline (Step by Step)

**Step 1 — Object Detection (YOLOv8)**

The image is passed through YOLOv8n (nano variant), a single-stage real-time object detector pretrained on the COCO dataset. It predicts bounding boxes and class labels in one forward pass. Detections are filtered to furniture-relevant classes (chair, couch, bed, dining table, plant, vase, etc.) with a confidence threshold of 0.35. Each detection includes a label, confidence score, and pixel coordinates.

**Step 2 — Feature Extraction (ResNet50, Transfer Learning)**

The full image is passed through ResNet50 pretrained on ImageNet. The final classification head is removed, and the output of the global average pooling layer is used as a 2048-dimensional visual embedding. This embedding captures textures, colors, and shapes without any fine-tuning — the ImageNet weights generalize well to room style recognition.

**Step 3 — Dimensionality Reduction (PCA)**

The 2048-dim ResNet50 embedding is projected down to 32 dimensions using Principal Component Analysis. PCA finds the axes of maximum variance in the training embedding space, discarding low-variance noise dimensions. This reduces computational cost for the downstream classifiers by 64x and improves kNN distance quality in lower-dimensional space.

**Step 4 — Style Classification (kNN + SVM)**

Two classifiers run on the 32-dim vector:

- kNN (k=5): finds the five nearest training embeddings in Euclidean space and returns the majority style label
- SVM (RBF kernel): finds maximum-margin class boundaries and returns calibrated probability scores via Platt scaling

The kNN prediction is used as the style label. The SVM probability for that class is used as the confidence score.

**Step 5 — Furniture Recommendation (Cosine Similarity)**

The room's 32-dim PCA vector is compared against a 42-item furniture database using cosine similarity. Items are ranked by their visual similarity score. The top 6 items are returned, prioritizing furniture that matches the detected style.

---

## Training the Classifiers (PCA, kNN, SVM)

The three sklearn models are trained on real room images from the **MIT Indoor Scenes dataset** (~2.4 GB, 15,620 images across 67 categories). Training is fully automated.

### What the training script does

1. Downloads the MIT Indoor Scenes dataset
2. Extracts the archive and deletes the `.tar` file to save disk space
3. Maps 67 MIT categories to the 6 project styles (Minimalist, Modern, Bohemian, Industrial, Scandinavian, Traditional)
4. Runs all images through ResNet50 using Apple MPS (M-series GPU acceleration) in batches of 64
5. Fits PCA on the resulting 2048-dim embeddings
6. Trains kNN and SVM on the 32-dim PCA-reduced embeddings
7. Saves `pca.pkl`, `knn.pkl`, `svm.pkl`
8. Recomputes furniture feature vectors in the same PCA coordinate space
9. Deletes the dataset images — they are not needed at runtime

After training, total disk usage is approximately 5 MB (three `.pkl` files). The 2.4 GB of images are fully cleaned up.

### Category to Style Mapping

| Style | MIT Indoor Categories |
|---|---|
| Minimalist | bathroom, closet, corridor, lobby, laundromat, hospitalroom, waitingroom |
| Modern | office, computerroom, meeting_room, movietheater, elevator, tv_studio, auditorium, gymnasium |
| Bohemian | children_room, art_studio, playroom, museum, library, bookstore, toystore, clothingstore, florist |
| Industrial | warehouse, garage, subway, trainstation, engine_room, stairscase, locker_room |
| Scandinavian | bedroom, living_room, dining_room, kitchen, pantry, nursery |
| Traditional | church_inside, ballroom, concert_hall, restaurant, bar, casino, classroom, lecture_theatre, hairsalon, mall, winecellar |

### Expected Training Time (Apple M4)

| Step | Time |
|---|---|
| Download (~2.4 GB) | 5-15 min |
| ResNet50 feature extraction (MPS) | 15-20 min |
| PCA + kNN + SVM training | 3-8 min |
| Total | 25-45 min |

---

## ML Models

| Model | Role | Pretrained | Source |
|---|---|---|---|
| YOLOv8n | Furniture object detection + bounding boxes | Yes (COCO) | Ultralytics |
| ResNet50 | 2048-dim visual feature extraction | Yes (ImageNet V1) | torchvision |
| PCA | 2048 to 32 dimensionality reduction | Trained on MIT Indoor Scenes | sklearn |
| kNN (k=5) | Primary room style classifier | Trained on MIT Indoor Scenes | sklearn |
| SVM (RBF) | Secondary classifier + confidence score | Trained on MIT Indoor Scenes | sklearn |
| Cosine Similarity | Furniture recommendation ranking | N/A | scipy |

---

## Frontend

The frontend is a React + TypeScript application built with Vite.

**Layout:** Two-panel side-by-side layout. The left panel handles image upload and displays the room photo with bounding box overlays. The right panel displays analysis results.

**Bounding box overlay:** After analysis, furniture detections are rendered as SVG rectangles scaled from original image coordinates to the displayed image size. Each box has a label showing the class name and confidence.

**Dynamic accent colors:** When a room style is detected, a CSS variable (`--accent`) is updated on the document root. Every UI element that uses `var(--accent)` — buttons, bars, badges, borders, glows — changes color to match the detected style automatically.

| Style | Accent Color |
|---|---|
| Minimalist | #A8DADC (muted teal) |
| Modern | #E9C46A (warm gold) |
| Bohemian | #E76F51 (terracotta) |
| Industrial | #8D99AE (slate blue) |
| Scandinavian | #95D5B2 (sage green) |
| Traditional | #C77DFF (soft violet) |

**Design system:** Dark glassmorphism aesthetic. All colors, spacing, radii, and shadows are defined as CSS custom properties. Inter font via Google Fonts.

---

## Project Structure

```
spac3d/
├── backend/
│   ├── main.py                  # FastAPI app — /health, /analyze
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── data/
│   │   └── furniture.json       # 42-item furniture recommendation database
│   ├── models/
│   │   ├── train_classifiers.py # Downloads MIT Indoor Scenes, trains PCA/kNN/SVM
│   │   ├── pca.pkl              # Generated after training
│   │   ├── knn.pkl              # Generated after training
│   │   └── svm.pkl              # Generated after training
│   └── utils/
│       ├── detection.py         # YOLOv8 inference + furniture class filtering
│       └── pipeline.py          # ResNet50, PCA, kNN, SVM, cosine similarity
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main two-panel layout + all components
│   │   ├── main.tsx             # React entry point
│   │   └── index.css            # Full dark glassmorphism design system
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
└── README.md
```

---

## Quick Start (Manual)

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train classifiers on MIT Indoor Scenes (first time only, ~30 min on M4)
python models/train_classifiers.py

# Start the server
uvicorn main:app --reload
```

Backend API: http://127.0.0.1:8000
Swagger docs: http://127.0.0.1:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

---

## Quick Start (Docker)

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000

---

## API Reference

### GET /health

Returns `{ "status": "ok" }` if the server is running and models are loaded.

### POST /analyze

Accepts a `multipart/form-data` image upload. Returns:

```json
{
  "status": "success",
  "detections": [
    { "label": "chair", "confidence": 0.91, "bbox": [120, 80, 320, 420] }
  ],
  "style": "Scandinavian",
  "confidence": 0.87,
  "recommendations": [
    {
      "id": "sca-001",
      "name": "Hans Wegner Style Chair",
      "style": "Scandinavian",
      "image_url": "https://picsum.photos/seed/chair5/400/300",
      "similarity_score": 0.923
    }
  ],
  "image_size": { "width": 1920, "height": 1080 }
}
```

---

## Notes

- **First startup without trained models:** If `pca.pkl`, `knn.pkl`, and `svm.pkl` are missing, `uvicorn main:app` will automatically trigger `train_classifiers.py` on the first request. This will download and process the full dataset. Run the script manually beforehand to avoid the delay.
- **YOLOv8 download:** The YOLOv8n weights (~6 MB) are downloaded automatically from the Ultralytics hub on first use.
- **ResNet50:** Uses torchvision ImageNet V1 pretrained weights (~100 MB), downloaded automatically from PyTorch hub on first use.
- **MPS acceleration:** Both training and inference use Apple MPS when available (M-series Macs). Falls back to CUDA, then CPU.
