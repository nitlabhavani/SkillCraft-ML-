import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Path to dataset
dataset_path = "dataset"

categories = ["cats", "dogs"]
data = []
labels = []

# Load images
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    label = categories.index(category)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            data.append(img.flatten())
            labels.append(label)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.3, random_state=42
)


model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100, "%")
print("Total Images:", len(data))
print("Cats:", labels.tolist().count(0))
print("Dogs:", labels.tolist().count(1))
