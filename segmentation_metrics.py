import os
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import jaccard_score

# Directories containing the prediction maps and ground truth masks
dir1 = '/home/wej36how/codes/CoSOD-main/result/Predictions/NWRDFRust_concatenated'
dir2 = '/home/wej36how/datasets/NWRDF/test/masks'

# Initialize lists to store scores
precisions = []
recalls = []
f1_scores = []
iou_scores = []

# Loop through all files in the prediction directory
for filename in os.listdir(dir1):
    pred_path = os.path.join(dir1, filename)
    gt_path = os.path.join(dir2, filename)
    print(pred_path)
    # Ensure that the file exists in both directories
    if os.path.exists(pred_path) and os.path.exists(gt_path):
        # Load the prediction and ground truth images
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Flatten the images to 1D arrays
        pred_flat = pred_img.flatten()
        gt_flat = gt_img.flatten()
        
        # Binarize the images (assuming binary segmentation masks)
        pred_flat = (pred_flat > 127).astype(np.uint8)
        gt_flat = (gt_flat > 127).astype(np.uint8)
        
        # Calculate precision, recall, and F1 score
        precision = precision_score(gt_flat, pred_flat)
        recall = recall_score(gt_flat, pred_flat)
        f1 = f1_score(gt_flat, pred_flat)
        iou = jaccard_score(gt_flat, pred_flat)
        
        # Append the scores to the lists
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        iou_scores.append(iou)



# Calculate average scores
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1_score = np.mean(f1_scores)
avg_iou = np.mean(iou_scores)

# Print the results
print(f'Average Precision: {avg_precision:.4f}')
print(f'Average Recall: {avg_recall:.4f}')
print(f'Average F1 Score: {avg_f1_score:.4f}')
print(f'Average iou Score: {avg_iou:.4f}')
