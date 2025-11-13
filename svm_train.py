import os
import joblib
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import time

# ----------------------------
# 1. Data Loading
# ----------------------------
def load_image_pair(a_path, b_path, label_path):
    """Read A (before), B (after), and label (change mask)."""
    A = cv2.imread(a_path)
    B = cv2.imread(b_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
    B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)
    
    # Convert to binary mask (0 or 1)
    label = (label > 127).astype(np.uint8)

    return A, B, label

# ----------------------------
# 2. Feature Extraction
# ----------------------------
def extract_pixel_features(A, B):
    """
    Extract richer features per pixel:
    - RGB from A and B
    - Absolute difference (|B - A|)
    - Gradient (Sobel) difference
    - Texture (local mean difference)

    Returns: (H, W, N_features)
    """

    diff = cv2.absdiff(B, A)
    grayA = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)

    # Gradient difference (Sobel)
    sobelA = cv2.Sobel(grayA, cv2.CV_64F, 1, 1, ksize=3)
    sobelB = cv2.Sobel(grayB, cv2.CV_64F, 1, 1, ksize=3)
    sobel_diff = cv2.absdiff(sobelA, sobelB)

    # Texture difference (local mean difference)
    meanA = cv2.blur(grayA, (5, 5))
    meanB = cv2.blur(grayB, (5, 5))
    mean_diff = cv2.absdiff(meanA, meanB)

    # Stack all channels
    feats = np.concatenate([
        A, B, diff,
        sobel_diff[..., np.newaxis],
        mean_diff[..., np.newaxis]
    ], axis=2)
    return feats

# ----------------------------
# 3. Training Data Preparation
# ----------------------------
def prepare_training_data(root_folder, num_samples=3000):
    X, y = [], []

    A_dir = os.path.join(root_folder, "A")
    B_dir = os.path.join(root_folder, "B")
    L_dir = os.path.join(root_folder, "label")
    
    
    image_names = os.listdir(A_dir)[:50]

    for name in image_names:
        A_path = os.path.join(A_dir, name)
        B_path = os.path.join(B_dir, name)
        L_path = os.path.join(L_dir, name)

        A, B, label = load_image_pair(A_path, B_path, L_path)
        feats = extract_pixel_features(A, B)

        H, W, C = feats.shape
        feats = feats.reshape(-1, C)
        labels = label.reshape(-1)

        # Randomly sample pixels
        idx = np.random.choice(len(labels), num_samples, replace=False)
        X.append(feats[idx])
        y.append(labels[idx])

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# ----------------------------
# 4. Balance Data
# ----------------------------
def balance_classes(X, y):
    X0 = X[y == 0]
    X1 = X[y == 1]

    if len(X1) == 0:
        return X, y  

    # Upsample the minority class (change pixels)
    X1_up = resample(X1, 
                     replace=True, 
                     n_samples=len(X0),  
                     random_state=42)
    y1_up = np.ones(len(X1_up)) # The labels for the upsampled data will be 1

    X_bal = np.vstack((X0, X1_up))
    y_bal = np.hstack((np.zeros(len(X0)), y1_up))

    print(f"Balanced data: {len(X0)} no-change, {len(X1_up)} change pixels")
    
    # Shuffle the balanced data
    shuffle_idx = np.random.permutation(len(X_bal))
    return X_bal[shuffle_idx], y_bal[shuffle_idx]

# ----------------------------
# 5. Train SVM
# ----------------------------
def train_svm(X, y):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        # class_weight='balanced' helps with initial class imbalance (pre-upsampling)
        ('svm', SVC(kernel='rbf', class_weight='balanced', random_state=42)) 
    ])

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return pipe

# ----------------------------
# 6. Predict Full Change Map
# ----------------------------
def predict_change_map(model, A, B, batch_size=20000):
    feats = extract_pixel_features(A, B)

    H, W, C = feats.shape
    X = feats.reshape(-1, C)

    preds = []
    
    for i in range(0, len(X), batch_size):
        preds.extend(model.predict(X[i:i+batch_size]))

    preds = np.array(preds).reshape(H, W)
    return preds

# ----------------------------
# 7. Post-process (clean prediction)
# ----------------------------
def postprocess(pred):
    # Ensure pred is in the correct format (binary, 0 or 1) for morphology
    pred = (pred * 255).astype(np.uint8) 
    
    kernel = np.ones((3, 3), np.uint8)
    
    # Opening: removes small objects/noise
    pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
    
    # Closing: fills small holes/gaps
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)

    # Convert back to 0/1 for consistency
    pred = (pred > 127).astype(np.uint8) 
    return pred

# ----------------------------
# 8. Visualization
# ----------------------------
def visualize_results(A, B, label, pred):
    fig, axs = plt.subplots(1, 4, figsize=(16, 6))

    axs[0].imshow(A)
    axs[0].set_title("Before (A)")

    axs[1].imshow(B)
    axs[1].set_title("After (B)")

    axs[2].imshow(label, cmap='gray')
    axs[2].set_title("Ground Truth")

    axs[3].imshow(pred, cmap='gray')
    axs[3].set_title("Predicted Change (v2)")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# ----------------------------
# 9. Main Script
# ----------------------------
if __name__ == "__main__": 
    
    train_path = r"LEVIR CD/train" 
    test_path = r"LEVIR CD/test"
    
    # --- Training Phase ---
    print("Preparing training data...")
    X, y = prepare_training_data(train_path, num_samples=2000)
    X_bal, y_bal = balance_classes(X, y)

    print("Training SVM...")
    model = train_svm(X_bal, y_bal) # Use the balanced data for training

    # Save model after training
    save_path = "svm_rbf_landchange_model.joblib"
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

    # --- Testing/Prediction Phase ---
    print("Testing on one sample image...")
 
    try:
        test_imgs = os.listdir(os.path.join(test_path, "A"))
        if not test_imgs:
            raise FileNotFoundError("No images found in the test 'A' directory.")
        sample = test_imgs[1]

        A_path = os.path.join(test_path, "A", sample)
        B_path = os.path.join(test_path, "B", sample)
        L_path = os.path.join(test_path, "label", sample)

        A, B, L = load_image_pair(A_path, B_path, L_path)

        start = time.time()
        pred = predict_change_map(model, A, B)
        print("Prediction done in", round(time.time() - start, 2), "seconds")

        pred = postprocess(pred)
        visualize_results(A, B, L, pred)

        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the paths and file structure are correct.")