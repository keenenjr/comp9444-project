
import os
import cv2
import numpy as np
import pandas as pd 
def load_train():
    train_data = [] # X_train
    train_labels = [] # X_test
    classes = 43
    train_path = os.getcwd() + "\dataset\Train"

    for i in os.listdir(train_path):
        dir = train_path + '/' + i
        if os.path.isdir(dir):
            for j in os.listdir(dir):
                try:
                    img_path = dir+ '/' +j
                    img = cv2.imread(img_path,-1)
                    # resize all images to 30,30
                    img = cv2.resize(img, (30,30), interpolation = cv2.INTER_NEAREST)
                    train_data.append(img)
                    train_labels.append(i)
                except Exception as e:
                    print(e)    
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    print(train_data.shape, train_labels.shape)
    return train_data, train_labels

def load_test():
    test_data = [] # y_train
    test_labels = [] # y_test
    test_path = os.getcwd() + "\dataset\Test"
    test_csv_path = os.getcwd() + "\dataset\Test.csv"
    test_df = pd.read_csv(test_csv_path, usecols=['ClassId', 'Path', 'Width', 'Height'])
    for index, row in test_df.iterrows():
        img_path = os.getcwd() + "\dataset/" + row['Path']
        img = cv2.imread(img_path, -1)
        img = cv2.resize(img, (30,30), interpolation=cv2.INTER_NEAREST)
        test_data.append(img)
        test_labels.append(row['ClassId'])

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    print(test_data.shape, test_labels.shape)
    return test_data, test_labels


# Example usage:
if __name__ == "__main__":
    India_train, India_label = load_test()
    print(India_train.shape, India_label.shape)