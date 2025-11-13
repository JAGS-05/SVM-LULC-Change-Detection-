import os
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import time

# ----------------------------
# Load Functions from Training
# ----------------------------
def load_image_pair(a_path, b_path, label_path):
    A = cv2.imread(a_path)
    B = cv2.imread(b_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
    B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)
    label = (label > 127).astype(np.uint8)
    return A, B, label


def extract_pixel_features(A, B):
    diff = cv2.absdiff(B, A)
    grayA = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)

    sobelA = cv2.Sobel(grayA, cv2.CV_64F, 1, 1, ksize=3)
    sobelB = cv2.Sobel(grayB, cv2.CV_64F, 1, 1, ksize=3)
    sobel_diff = cv2.absdiff(sobelA, sobelB)

    meanA = cv2.blur(grayA, (5, 5))
    meanB = cv2.blur(grayB, (5, 5))
    mean_diff = cv2.absdiff(meanA, meanB)

    feats = np.concatenate([
        A, B, diff,
        sobel_diff[..., np.newaxis],
        mean_diff[..., np.newaxis]
    ], axis=2)
    return feats


def predict_change_map(model, A, B, batch_size=20000):
    feats = extract_pixel_features(A, B)
    H, W, C = feats.shape
    X = feats.reshape(-1, C)

    preds = []
    for i in range(0, len(X), batch_size):
        preds.extend(model.predict(X[i:i+batch_size]))

    preds = np.array(preds).reshape(H, W)
    return preds


def postprocess(pred):
    pred = (pred * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
    pred = (pred > 127).astype(np.uint8)
    return pred


def visualize_results(A, B, label, pred):
    fig, axs = plt.subplots(1, 4, figsize=(16, 6))
    axs[0].imshow(A); axs[0].set_title("Before (A)")
    axs[1].imshow(B); axs[1].set_title("After (B)")
    axs[2].imshow(label, cmap='gray'); axs[2].set_title("Ground Truth")
    axs[3].imshow(pred, cmap='gray'); axs[3].set_title("Predicted Change")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main Testing Code
# ----------------------------
if __name__ == "__main__":
    model_path = "svm_rbf_landchange_model.joblib"  # saved model
    test_path = r"LEVIR CD/test"                

    model = joblib.load(model_path)
    print("Model loaded successfully.")

    test_imgs = os.listdir(os.path.join(test_path, "A"))
    print(f"Found {len(test_imgs)} test images.")

    total_y_true, total_y_pred = [], []

    for i, name in enumerate(test_imgs[1:3]):  
        print(f"\nTesting on {name}...")
        A_path = os.path.join(test_path, "A", name)
        B_path = os.path.join(test_path, "B", name)
        L_path = os.path.join(test_path, "label", name)

        A, B, L = load_image_pair(A_path, B_path, L_path)

        start = time.time()
        pred = predict_change_map(model, A, B)
        pred = postprocess(pred)
        print(f"Prediction done in {round(time.time()-start, 2)} sec")

        visualize_results(A, B, L, pred)

        # Flatten for metrics
        total_y_true.extend(L.flatten())
        total_y_pred.extend(pred.flatten())

    print("\n=== Overall Performance on Test Set ===")
    print(classification_report(total_y_true, total_y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(total_y_true, total_y_pred))
