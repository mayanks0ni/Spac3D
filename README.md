# Spac3D 🧊
**Pseudo-3D from Single Image Pipeline**

Spac3D models a mapping from 2D image space to 3D spatial representation using pretrained deep learning models, combining supervised learning, probabilistic inference, and geometric reconstruction. This provides a robust monocular scene understanding pipeline optimized for speed and accuracy.

## 🧠 Approach

1. **Depth Estimation**: We extract spatial geometry mappings by feeding a single 2D image into **MiDaS** (a supervised learning neural network solving the probabilistic problem of continuous depth map prediction).
2. **Segmentation**: Objects are identified and separated via region-grouping using the **Segment Anything Model (SAM)**.
3. **3D Reconstruction**: Using **Open3D**, depth maps and masks are converted via dimensionality transformation `f(x) \rightarrow (x, y, z)` into a structured 3D point cloud and mesh.
4. **Rendering**: The reconstructed environment is projected in a browser using **Three.js** and **React Three Fiber** for interactive visualization.

## ⚙️ Tech Stack

**Backend (Python / Machine Learning):**
- FastAPI
- MiDaS (Depth Estimation)
- Segment Anything Model (SAM)
- Open3D (Point Cloud & Mesh Reconstruction)

**Frontend (React / 3D Web Rendering):**
- Next.js (App Router, Vanilla CSS)
- Three.js
- React Three Fiber & Drei

## 🚀 Getting Started

### Local Setup (Backend)

1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Set up the virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Start the FastAPI development server:
   ```bash
   uvicorn main:app --reload
   ```
   *The backend will be running on `http://localhost:8000`*

### Local Setup (Frontend)

1. Navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
   *The UI will be accessible at `http://localhost:3000`*

## 📝 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
