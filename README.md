# 🏠 SPAC3D — AI Room Intelligence & Furniture Reconstruction

**SPAC3D** is a premium, full-stack AI platform designed to transform 2D room photography into actionable interior design data. By leveraging a state-of-the-art open-vocabulary ML pipeline, SPAC3D detects specific furniture elements, identifies architectural styles, and offers a direct bridge to global marketplaces via Amazon affiliate integration.

---

## 🚀 Key Features

### 🔍 Open-Vocabulary Intelligence
Unlike standard detectors limited to basic COCO classes, SPAC3D utilizes **YOLO-World**. This allows for the granular detection of interior-specific items including:
- **Major Furniture**: Sofas, Dining Tables, Beds, Cabinets.
- **Accents & Decor**: Cushions, Rugs, Vases, Lamps, Mirrors.
- **Architectural Details**: Shelves, Drawers, Sideboards.

### 🎨 Intelligent Style Recognition
A hybrid **ResNet50 + PCA + kNN/SVM** pipeline analyzes the global distribution of features to classify your room into one of six core design identities:
- 🌿 **Minimalist** | ✨ **Modern** | 🏺 **Bohemian**
- 🧱 **Industrial** | 🌲 **Scandinavian** | 🏛️ **Traditional**

### 🛒 "Shop the Look" Integration
Every detected object and identified style is instantly shoppable.
- **Contextual Search**: Click a detected "Cushion" to see matching recommendations in the current room style.
- **Amazon Market Bridge**: Direct "Shop on Amazon" integration for both individual items and full-room style palettes.

### 🌓 Dynamic Glassmorphism UI
A high-performance React frontend featuring:
- **Real-time Accent Sync**: The UI theme (glows, borders, buttons) automatically shifts colors to match the detected room style.
- **Interactive Bounding Boxes**: Hover and click functionality for filtered recommendations.
- **Mobile Responsive**: Built with Vite and native CSS for zero-latency interaction.

---

## 🧠 The ML Architecture

| Stage | Technology | Detail |
|---|---|---|
| **Detection** | `YOLO-World` | Open-vocabulary inference for granular interior details. |
| **Extraction** | `ResNet50` | Global average pooling for 2048-dim visual feature extraction. |
| **Reduction** | `PCA` | Dimensionality reduction (2048 → 32) for optimized similarity. |
| **Style** | `SVM + kNN` | Multi-classifier approach for high-confidence style labeling. |
| **Recs** | `Cosine Similarity` | Vector-based similarity search against a curated furniture catalog. |

---

## 📁 Project Structure

```bash
spac3d/
├── backend/
│   ├── main.py              # FastAPI Entry Point
│   ├── models/
│   │   ├── fast_train.py    # High-speed kNN/SVM trainer
│   │   └── *.pkl            # Serialized ML artifacts
│   ├── utils/
│   │   ├── detection.py     # YOLO-World & IoA Suppression logic
│   │   └── pipeline.py      # Feature extraction & recommendation engine
│   └── data/
│       └── furniture.json   # Curated furniture database
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Interactive UI & State management
│   │   └── index.css        # Glassmorphism Design System
│   └── package.json         # Node dependencies
├── add_decor.py             # Utility: Style-based data generator
├── docker-compose.yml       # Full Orchestration
└── README.md
```

---

## 🛠️ Quick Start

### 1. Docker (Recommended)
The fastest way to get SPAC3D running with all dependencies pre-configured.
```bash
docker-compose up --build
```
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000

### 2. Manual Setup

#### Backend
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🔧 Deployment Notes

- **Model Download**: On first run, the system will automatically download `yolov8s-world.pt` (~45MB) and `ResNet50` weights.
- **CPU vs GPU**: The pipeline is optimized to run on CPU by default to ensure stability in containerized environments, but supports CUDA/MPS for accelerated inference.
- **Dataset**: Classifiers are trained on the **MIT Indoor Scenes** dataset, mapped into the 6 project styles for maximum accuracy in interior environments.

---

## 📜 License
*Proprietary — Developed for SPAC3D Interior Intelligence.*
