import os
import io
import json
import pickle
import shutil
import tarfile
import urllib.request
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError

import torch
import torchvision.transforms as T
from torchvision import models

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

STYLES = ["Minimalist", "Modern", "Bohemian", "Industrial", "Scandinavian", "Traditional"]
PCA_COMPONENTS = 32
KNN_K = 5
RANDOM_SEED = 42
BATCH_SIZE = 64
MAX_PER_STYLE = 500

MODELS_DIR   = Path(__file__).parent
DATASET_DIR  = MODELS_DIR / "mit_indoor"
ARCHIVE_PATH = MODELS_DIR / "indoorCVPR_09.tar"
IMAGES_DIR   = DATASET_DIR / "Images"

PCA_PATH       = MODELS_DIR / "pca.pkl"
KNN_PATH       = MODELS_DIR / "knn.pkl"
SVM_PATH       = MODELS_DIR / "svm.pkl"
FURNITURE_PATH = MODELS_DIR.parent / "data" / "furniture.json"

DATASET_URL = (
    "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
)

CATEGORY_TO_STYLE: dict[str, str] = {
    "bathroom":        "Minimalist",
    "closet":          "Minimalist",
    "corridor":        "Minimalist",
    "lobby":           "Minimalist",
    "laundromat":      "Minimalist",
    "hospitalroom":    "Minimalist",
    "waitingroom":     "Minimalist",

    "office":          "Modern",
    "computerroom":    "Modern",
    "meeting_room":    "Modern",
    "movietheater":    "Modern",
    "elevator":        "Modern",
    "tv_studio":       "Modern",
    "auditorium":      "Modern",
    "gymnasium":       "Modern",

    "children_room":   "Bohemian",
    "art_studio":      "Bohemian",
    "playroom":        "Bohemian",
    "museum":          "Bohemian",
    "library":         "Bohemian",
    "bookstore":       "Bohemian",
    "toystore":        "Bohemian",
    "clothingstore":   "Bohemian",
    "florist":         "Bohemian",

    "warehouse":       "Industrial",
    "garage":          "Industrial",
    "subway":          "Industrial",
    "trainstation":    "Industrial",
    "engine_room":     "Industrial",
    "stairscase":      "Industrial",
    "locker_room":     "Industrial",

    "bedroom":         "Scandinavian",
    "living_room":     "Scandinavian",
    "dining_room":     "Scandinavian",
    "kitchen":         "Scandinavian",
    "pantry":          "Scandinavian",
    "nursery":         "Scandinavian",
    "inside_bus":      "Scandinavian",

    "church_inside":   "Traditional",
    "ballroom":        "Traditional",
    "concert_hall":    "Traditional",
    "restaurant":      "Traditional",
    "bar":             "Traditional",
    "casino":          "Traditional",
    "prisoncell":      "Traditional",
    "classroom":       "Traditional",
    "lecture_theatre": "Traditional",
    "operating_room":  "Traditional",
    "hairsalon":       "Traditional",
    "jewelleryshop":   "Traditional",
    "mall":            "Traditional",
    "videostore":      "Traditional",
    "shoeshop":        "Traditional",
    "spa":             "Traditional",
    "winecellar":      "Traditional",
}


def _progress_hook(count: int, block_size: int, total_size: int) -> None:
    downloaded = count * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1024 / 1024
        print(f"\r  {pct:5.1f}%  {mb:7.1f} MB downloaded", end="", flush=True)


def download_dataset() -> None:
    if IMAGES_DIR.exists() and any(IMAGES_DIR.iterdir()):
        print(f"[dataset] Found existing images at {IMAGES_DIR} — skipping download.")
        return

    print(f"[dataset] Downloading MIT Indoor Scenes (~2.4 GB) from:\n  {DATASET_URL}")
    print("  This will take 5-15 minutes depending on your connection speed...")
    urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH, reporthook=_progress_hook)
    print()

    print(f"[dataset] Extracting archive to {DATASET_DIR} ...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE_PATH, "r") as tar:
        tar.extractall(DATASET_DIR)
    print("[dataset] Extraction complete.")

    ARCHIVE_PATH.unlink(missing_ok=True)
    print("[dataset] Archive removed.")


def collect_image_paths() -> tuple[list[Path], list[int]]:
    style_to_idx = {s.lower(): i for i, s in enumerate(STYLES)}
    per_style_count = [0] * len(STYLES)
    paths, labels = [], []

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found at {IMAGES_DIR}.")

    for category_dir in sorted(IMAGES_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        cat = category_dir.name.lower()
        style = CATEGORY_TO_STYLE.get(cat)
        if style is None:
            continue
        idx = style_to_idx[style.lower()]

        for img_path in sorted(category_dir.iterdir()):
            if per_style_count[idx] >= MAX_PER_STYLE:
                break
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            paths.append(img_path)
            labels.append(idx)
            per_style_count[idx] += 1

    for i, s in enumerate(STYLES):
        print(f"  {s:15s}: {per_style_count[i]} images")

    if not paths:
        raise RuntimeError("No images collected. Check IMAGES_DIR contains category subfolders.")

    return paths, labels


_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("[device] Apple MPS detected — using MPS.")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("[device] CUDA detected — using CUDA.")
        return torch.device("cuda")
    print("[device] No GPU found — using CPU.")
    return torch.device("cpu")


def extract_features(
    paths: list[Path],
    labels: list[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    # Transfer Learning: ResNet50 pretrained on ImageNet as feature extractor
    print("[extract] Loading ResNet50 (ImageNet V1 weights)...")
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(backbone.children())[:-1])
    model.eval().to(device)

    n = len(paths)
    X_list, y_list = [], []
    failed = 0

    print(f"[extract] Extracting features from {n} images in batches of {BATCH_SIZE}...")
    for batch_start in range(0, n, BATCH_SIZE):
        batch_paths  = paths[batch_start : batch_start + BATCH_SIZE]
        batch_labels = labels[batch_start : batch_start + BATCH_SIZE]

        tensors, valid_labels = [], []
        for p, lbl in zip(batch_paths, batch_labels):
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(_TRANSFORM(img))
                valid_labels.append(lbl)
            except (UnidentifiedImageError, OSError):
                failed += 1
                continue

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model(batch_tensor)
        feats = feats.squeeze(-1).squeeze(-1)

        X_list.append(feats.cpu().float().numpy())
        y_list.extend(valid_labels)

        done = min(batch_start + BATCH_SIZE, n)
        print(f"\r  {done}/{n} images processed ({done/n*100:.1f}%)", end="", flush=True)

    print()
    if failed:
        print(f"[extract] Skipped {failed} unreadable images.")

    X = np.vstack(X_list).astype(np.float64)
    y = np.array(y_list, dtype=int)
    print(f"[extract] Feature matrix: {X.shape[0]} x {X.shape[1]} (float64)")
    return X, y


def train_and_save() -> None:
    download_dataset()

    print("\n[collect] Collecting image paths by style...")
    paths, labels = collect_image_paths()
    print(f"[collect] Total: {len(paths)} images across {len(STYLES)} styles")

    device = _get_device()
    X, y = extract_features(paths, labels, device)

    # PCA: dimensionality reduction 2048 -> 32
    print(f"\n[pca] Fitting PCA (2048 -> {PCA_COMPONENTS} dims) on {X.shape[0]} embeddings...")
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full", random_state=RANDOM_SEED)
    X_reduced = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    print(f"[pca] Explained variance: {explained:.1%}")

    # kNN: primary style classifier
    print(f"[knn] Training kNN (k={KNN_K}) on {X_reduced.shape[0]} samples...")
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=KNN_K, metric="euclidean", n_jobs=-1)),
    ])
    knn_pipeline.fit(X_reduced, y)
    print(f"[knn] Training accuracy: {knn_pipeline.score(X_reduced, y):.1%}")

    # SVM: secondary classifier with calibrated probability scores
    print("[svm] Training SVM (RBF kernel, probability=True)...")
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale",
                    probability=True, random_state=RANDOM_SEED)),
    ])
    svm_pipeline.fit(X_reduced, y)
    print(f"[svm] Training accuracy: {svm_pipeline.score(X_reduced, y):.1%}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PCA_PATH, "wb") as f: pickle.dump(pca, f)
    with open(KNN_PATH, "wb") as f: pickle.dump(knn_pipeline, f)
    with open(SVM_PATH, "wb") as f: pickle.dump(svm_pipeline, f)
    print(f"\n[save] Models saved: pca.pkl, knn.pkl, svm.pkl -> {MODELS_DIR}")

    # Cosine Similarity: recompute furniture feature vectors in real PCA space
    print("\n[furniture] Regenerating furniture feature vectors using real PCA space...")
    style_to_idx = {s.lower(): i for i, s in enumerate(STYLES)}

    style_centroids = np.zeros((len(STYLES), PCA_COMPONENTS), dtype=np.float64)
    for i in range(len(STYLES)):
        mask = y == i
        if mask.sum() > 0:
            style_centroids[i] = X_reduced[mask].mean(axis=0)

    item_rng = np.random.RandomState(777)
    with open(FURNITURE_PATH, "r") as f:
        furniture = json.load(f)

    for item in furniture:
        idx = style_to_idx.get(item["style"].lower(), 0)
        vec = style_centroids[idx] + item_rng.normal(0, 0.05, PCA_COMPONENTS)
        if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
            vec = np.zeros(PCA_COMPONENTS, dtype=np.float64)
        item["feature_vector"] = [round(float(v), 6) for v in vec]

    with open(FURNITURE_PATH, "w") as f:
        json.dump(furniture, f, indent=2)

    print(f"[furniture] Updated {len(furniture)} items -> {FURNITURE_PATH}")

    if DATASET_DIR.exists():
        print(f"\n[cleanup] Deleting dataset images ({DATASET_DIR})...")
        shutil.rmtree(DATASET_DIR)
        print("[cleanup] Dataset images deleted.")

    print("\nTraining complete. Run `uvicorn main:app --reload` to start the server.")


if __name__ == "__main__":
    train_and_save()
