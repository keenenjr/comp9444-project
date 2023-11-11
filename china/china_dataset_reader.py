import cv2
import numpy as np
import os
import pandas as pd

def load_traffic_sign_dataset():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, 'annotations.csv')
    print("CSV Path:", csv_path)


    data = pd.read_csv(csv_path)
    image_fnames = data['file_name']
    categories = data['category']

    def load_traffic_sign_image(fname):
        img = cv2.imread(fname, -1)
        img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_NEAREST)
        return img

    images = np.array([load_traffic_sign_image(os.path.join(base_dir, 'images', fn)) for fn in image_fnames])

    return images, categories
