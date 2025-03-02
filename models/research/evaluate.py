#
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util, visualization_utils as viz_utils
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt

# Path configurations
label_map_path = "C:/Github-Project_file/Comvision/Lab/models/ssd_mobilenet_v2/train/label_map.pbtxt"
val_record_path = "C:/Github-Project_file/Comvision/Lab/models/ssd_mobilenet_v2/train/val.record"
model_dir = r"C:\Github-Project_file\Comvision\Lab\exported_model\saved_model"

# Load the label map
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Define a function to parse TFRecord
def parse_tfrecord_fn(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.image.decode_jpeg(example['image/encoded'])
    label = tf.sparse.to_dense(example['image/object/class/label'])
    return image, label



# Load and parse the validation dataset
raw_dataset = tf.data.TFRecordDataset(val_record_path)
parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

# Load the saved model
detect_fn = tf.saved_model.load(model_dir)


# Initialize lists for true and predicted labels
true_labels = []
predicted_labels = []
predicted_scores = []



# Run inference on the validation dataset
confidence_threshold = 0.5
for image, labels in parsed_dataset:
    input_tensor = tf.expand_dims(image, 0)
    detections = detect_fn(input_tensor)

    # Get the highest scoring detection
    classes = detections["detection_classes"][0].numpy().astype(np.int32)
    scores = detections["detection_scores"][0].numpy()

    # Use confidence threshold to filter predictions
    if scores[0] >= confidence_threshold:
        predicted_labels.append(classes[0])  # Predicted class
        predicted_scores.append(scores[0])  # Predicted confidence
    else:
        predicted_labels.append(0)  # No detection
        predicted_scores.append(0)  # No confidence

    # Use the first ground truth label (assuming one object per image)
    true_labels.append(labels.numpy()[0] if len(labels.numpy()) > 0 else 0)

# Convert lists to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)
predicted_scores = np.array(predicted_scores)

# Compute evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

# Compute mAP (Mean Average Precision)
try:
    map_score = average_precision_score(true_labels, predicted_scores, average='macro')
except ValueError:
    map_score = 0  # Handle case where AP calculation fails due to class imbalance

# print("******************************")
# print('adadadadadsadadadasdad', parsed_dataset)
# print(true_labels)
# print(predicted_labels)
# print(precision_score)
# print("******************************")

print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")
print(f"mAP: {map_score:.2%}")