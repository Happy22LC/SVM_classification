"""
I kept all the dataset on my project folder
 To run the program:
  python tb_hog_svm.py --data_dir "/path/to/TB_Chest_Radiography_Database" --feature hog --output_dir outputs
Requires:
  pip install numpy scikit-image scikit-learn matplotlib tqdm joblib
"""

import os
import json
import argparse
import warnings
from glob import glob
from datetime import datetime

import numpy as np
from tqdm import tqdm

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog, local_binary_pattern
from skimage import exposure

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.svm import SVC
from joblib import dump
import matplotlib
matplotlib.use("Agg")  # safe for headless runs; remove if you want interactive
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")



#  Data Preprocessing and Feature Extraction

def find_image_files(folder):
    # return a list of image file paths in 'folder' with common extensions, and it will keep the loader robust to mixed image types
    image_extension = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    return [f for f in glob(os.path.join(folder, "*")) if f.lower().endswith(image_extension)]


def read_and_basic_process(path, img_size=(128, 128), grayscale=True):

    #need grayscale because Color is not essential for X-ray texture/edge patterns and grayscale reduces dimensionality and noise for classical ML.
    # read image, convert to grayscale, resize to img_size
    img = imread(path)
    # convert to grayscale if needed, then resize to a consistent size for feature extraction.
    if grayscale:
        if img.ndim == 2:
            gray = img   # already grayscale
        else:
            gray = rgb2gray(img)
        img_resized = resize(gray, img_size, anti_aliasing=True)
        return img_resized.astype(np.float32)
    else:
        img_resized = resize(img, img_size, anti_aliasing=True, preserve_range=False)
        return img_resized.astype(np.float32)



# Feature extraction (HOG recommended, LBP optional) : Step 2

def extract_hog_feature(img_gray,
                        orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2)):
    #Compute Histogram of Oriented Gradients (HOG) for a single grayscale image.
    #HOG captures edge or shape that are informative for X-rays.
    # Returns a fixed length 1D feature vector that is suitable for SVM.

    feat = hog(
        img_gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True,    # gamma normalization helps with illumination variations
        feature_vector=True
    )
    return feat


def extract_lbp_feature(img_gray, p=8, r=1):

    #Compute a uniform Local Binary Patterns histogram feature for a grayscale image.
    #LBP encodes local texture patterns.
    # here, p : number of circular neighbors r:radius of the circular neighborhood

    lbp = local_binary_pattern(img_gray, p, r, method="uniform")
    n_bins = p + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    
    return hist.astype(np.float32)

# loop over the paths preprocess each image, and stack features into X.
def build_feature_matrix(paths, feature="hog", img_size=(128, 128),
                         hog_orient=9, hog_pix=(8, 8), hog_cell=(2, 2),
                         lbp_p=8, lbp_r=1):
    feats = []
    for path in tqdm(paths, desc=f"Extracting {feature.upper()}"):
        try:
            img = read_and_basic_process(path, img_size=img_size, grayscale=True)
            if feature == "hog":
                f = extract_hog_feature(
                    img, orientations=hog_orient,
                    pixels_per_cell=hog_pix, cells_per_block=hog_cell
                )
            elif feature == "lbp":
                f = extract_lbp_feature(img, p=lbp_p, r=lbp_r)
            else:
                raise ValueError("Unsupported feature type. Use 'hog' or 'lbp'.")
            feats.append(f)
        except Exception as e:
            # skip unreadable or corrupt images without breaking the whole run.
            print(f"[WARN] Skipped {path}: {e}")
    feats = np.vstack(feats)
    return feats


def visualize_hog(img_path, out_png, img_size=(128, 128),
                  hog_orient=9, hog_pix=(8, 8), hog_cell=(2, 2)):
    #save a side-by-side visualization of the preprocessed grayscale input
    # the HOG "edge energy" image for the report narrative
    img = read_and_basic_process(img_path, img_size=img_size, grayscale=True)
    fd, hog_image = hog(
        img,
        orientations=hog_orient,
        pixels_per_cell=hog_pix,
        cells_per_block=hog_cell,
        block_norm="L2-Hys",
        transform_sqrt=True,
        visualize=True,
        feature_vector=True
    )
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, hog_image.max()))
    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img, cmap="gray")
    ax1.set_title("Input (grayscale)")
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(hog_image_rescaled, cmap="gray")
    ax2.set_title("HOG visualization")
    ax2.axis("off")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)



# Training & evaluation (Steps 3–5)

# split the data, run GridSearchCV to tune SVM, evaluate on test set,
# and save model, confusion matrix figure, text or JSON reports
def train_and_evaluate(x, y, output_dir, seed=42, test_size=0.3,
                       kernel="rbf", scoring="f1_macro", cv_splits=5):

    """
    Arguments:
        X, y          : features and labels
        kernel        : 'rbf' (recommended), 'linear', or 'poly'
        scoring       : model selection metric for GridSearchCV
        cv_splits     : number of StratifiedKFold splits during tuning
    """
    # Data Splitting

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )


    # SVM Model Training
    # standardize features then SVM. with_mean=False works well even if features look
    svc = SVC(kernel=kernel, class_weight="balanced")
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # safe for large dense or sparse-ish features
        ("svc", svc)
    ])

    # grid based on kernel
    # parameter grid per kernel tune C and gamma for RBF, only C for linear and C/gamma/degree for poly.
    if kernel == "rbf":
        param_grid = {
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", 1e-2, 1e-3, 1e-4],
        }
    elif kernel == "linear":
        param_grid = {"svc__C": [0.1, 1, 10, 100]}
    elif kernel == "poly":
        param_grid = {
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", 1e-3, 1e-4],
            "svc__degree": [2, 3]
        }
    else:
        raise ValueError("Unsupported kernel. Use rbf|linear|poly.")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=scoring, # choose macro-F1 to balance precision
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(x_train, y_train)

    # Evaluation on the est set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Normal", "TB"], digits=4)

    # save confusion matrix figure
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "TB"])
    disp.plot(ax=ax, values_format="d", colorbar=False)
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "figures", "cm_test.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # save trained model
    model_path = os.path.join(output_dir, "models", "svm_best.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(best_model, model_path)

    # save report text + JSON summary
    report_path = os.path.join(output_dir, "reports", "classification_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("Best Params:\n")
        f.write(json.dumps(grid.best_params_, indent=2))
        f.write("\n\nCross-validated best score ({}): {:.4f}\n".format(scoring, grid.best_score_))
        f.write("\nTest Accuracy: {:.4f}\n".format(acc))
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    # also dump a small JSON summary
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "best_params": grid.best_params_,
        "cv_best_score": float(grid.best_score_),
        "scoring": scoring,
        "test_accuracy": acc,
        "confusion_matrix": cm.tolist(),
    }
    summary_path = os.path.join(output_dir, "reports", "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # console recap
    print("\n=== RESULTS ===")
    print("Best params:", grid.best_params_)
    print(f"Best CV score ({scoring}): {grid.best_score_:.4f}")
    print("Test Accuracy:", f"{acc:.4f}")
    print("\nClassification Report:\n", report)
    print(f"\nSaved: {cm_path}\nSaved: {model_path}\nSaved: {report_path}\nSaved: {summary_path}")

    return best_model, acc, cm, report, grid.best_params_



# Main (Step 1 + Step 2 + Step 3–6)
#
def main():
    parser = argparse.ArgumentParser(description="TB Chest X-ray SVM (HOG/LBP features)")

    # paths and data
    parser.add_argument("--data_dir", required=True,
                        help="Folder containing Normal/ and Tuberculosis/ subfolders.")
    parser.add_argument("--feature", default="hog", choices=["hog", "lbp"],
                        help="Feature extractor to use (default: hog).")

    # feature choice and preprocessing
    parser.add_argument("--img_size", type=int, default=64,
                        help="Resize to IMG_SIZE x IMG_SIZE (default: 64).")

    # splits and randomness
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--test_size", type=float, default=0.30, help="Test size fraction (default: 0.30).")

    # SVM and tuning
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "linear", "poly"],
                        help="SVM kernel (default: rbf).")
    parser.add_argument("--scoring", default="f1_macro",
                        help="GridSearchCV scoring (e.g., f1_macro, balanced_accuracy, roc_auc_ovr).")
    parser.add_argument("--cv", type=int, default=5, help="Stratified CV splits (default: 5).")

    # outputs and visualisation
    parser.add_argument("--output_dir", default="outputs", help="Where to save models/figures/reports.")
    parser.add_argument("--save_hog_viz", action="store_true",
                        help="Save a few HOG visualizations (only if feature=hog).")
    args = parser.parse_args()

    # validate folders
    # expect exactly two class folders: Normal/ and Tuberculosis/ (case-sensitive here).

    normal_dir = os.path.join(args.data_dir, "Normal")
    tb_dir = os.path.join(args.data_dir, "Tuberculosis")
    if not (os.path.isdir(normal_dir) and os.path.isdir(tb_dir)):
        raise SystemExit(f"[ERROR] Expected Normal/ and Tuberculosis/ inside: {args.data_dir}")

    # collect image paths and built labels: 0=Normal, 1=TB
    normal_paths = sorted(find_image_files(normal_dir))
    tb_paths = sorted(find_image_files(tb_dir))
    X_paths = normal_paths + tb_paths
    y = np.array([0] * len(normal_paths) + [1] * len(tb_paths), dtype=np.int64)

    print(f"Found {len(normal_paths)} Normal and {len(tb_paths)} TB images (total={len(X_paths)}).")

    # Build features:Step 2: preprocess and Feature Extraction

    IMG_SIZE = (args.img_size, args.img_size)
    if args.feature == "hog":
        X = build_feature_matrix(
            X_paths, feature="hog", img_size=IMG_SIZE,
            hog_orient=9, hog_pix=(8, 8), hog_cell=(2, 2)
        )
    else:
        X = build_feature_matrix(
            X_paths, feature="lbp", img_size=IMG_SIZE,
            lbp_p=8, lbp_r=1
        )

    # Train + evaluate
    # steps 3–5: Train, Tune, Evaluate and Step 6: Save reports
    os.makedirs(args.output_dir, exist_ok=True)
    best_model, acc, cm, report, best_params = train_and_evaluate(
        X, y, output_dir=args.output_dir, seed=args.seed, test_size=args.test_size,
        kernel=args.kernel, scoring=args.scoring, cv_splits=args.cv
    )

    # save HOG visualizations for  first few images that helps the report
    if args.save_hog_viz and args.feature == "hog":
        figdir = os.path.join(args.output_dir, "figures")
        os.makedirs(figdir, exist_ok=True)
        for p in X_paths[:3]:
            out_png = os.path.join(figdir, f"hog_{os.path.basename(p)}.png")
            try:
                visualize_hog(p, out_png, img_size=IMG_SIZE)
                print("Saved HOG viz:", out_png)
            except Exception as e:
                print(f"[WARN] HOG viz failed for {p}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
