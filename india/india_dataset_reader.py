import os
import cv2
import numpy as np

def load_india_dataset():
    train_data = []  # X_train
    train_labels = []  # X_test
    classes = 46  # Including 0 to 45
    dataset_path = os.path.join(os.getcwd(), "india", "Images")

    for label in range(classes):
        label_path = os.path.join(dataset_path, str(label))
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                try:
                    img_path = os.path.join(label_path, img_name)
                    img = cv2.imread(img_path)
                    
                    # Convert to RGB if the image has a single channel (grayscale)
                    if img.shape[-1] == 1:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    
                    # Resize all images to 30x30
                    img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_NEAREST)
                    
                    train_data.append(img)
                    train_labels.append(label)
                except Exception as e:
                    print(e)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    return train_data, train_labels

# Example usage:
if __name__ == "__main__":
    India_train, India_label = load_india_dataset()
    print(India_train.shape, India_label.shape)

