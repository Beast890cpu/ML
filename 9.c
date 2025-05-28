import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.datasets import fetch_olivetti_faces 
data = fetch_olivetti_faces() 
  
data.keys() 
  
dict_keys(['data', 'images', 'target', 'DESCR']) 
  
print("Data Shape:", data.data.shape) 
print("Target Shape:", data.target.shape) 
print("There are {} unique persons in the 
dataset".format(len(np.unique(data.target)))) 
print("Size of each image is 
{}x{}".format(data.images.shape[1],data.images.shape[1]))

  def print_faces(images, target, top_n): 

top_n = min(top_n, len(images)) 
 
grid_size = int(np.ceil(np.sqrt(top_n))) 
fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15)) 

   for i, ax in enumerate(axes.ravel()): 
        if i < top_n: 
            ax.imshow(images[i], cmap='bone') 
            ax.axis('off') 
            ax.text(2, 12, str(target[i]), fontsize=9, color='red') 
            ax.text(2, 55, f"face: {i}", fontsize=9, color='blue') 
        else: 
            ax.axis('off') 
 
    plt.show() 

  
def display_unique_faces(pics): 
    fig = plt.figure(figsize=(24, 10))  # Set figure size 
    columns, rows = 10, 4  # Define grid dimensions 
  
    for i in range(1, columns * rows + 1): 
        img_index = 10 * i - 1  # Calculate the image index 
        if img_index < pics.shape[0]:  # Check for valid image index 
            img = pics[img_index, :, :] 
            ax = fig.add_subplot(rows, columns, i) 
            ax.imshow(img, cmap='gray') 
            ax.set_title(f"Person {i}", fontsize=14) 
            ax.axis('off') 
 
    plt.suptitle("There are 40 distinct persons in the dataset", fontsize=24) 
    plt.show() 
display_unique_faces(data.images)

  X = data.data 
Y = data.target 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, 
random_state=46) 
 
print("x_train: ",x_train.shape) 
print("x_test: ",x_test.shape) 
x_train:  (280, 4096) 
x_test:  (120, 4096)

   
nb = GaussianNB() 
nb.fit(x_train, y_train) 

y_pred = nb.predict(x_test) 
  
nb_accuracy = round(accuracy_score(y_test, y_pred) * 100, 2) 

cm = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix:") 
print(cm) 

print(f"Naive Bayes Accuracy: {nb_accuracy}%")

  nb.fit(x_train, y_train) 
 
y_pred = nb.predict(x_test) 

accuracy = round(accuracy_score(y_test, y_pred) * 100, 2) 
print(f"Multinomial Naive Bayes Accuracy: {accuracy}%") 

  print(f"Number of misclassified images: {num_misclassified}") 
print(f"Total images in test set: {len(y_test)}") 
print(f"Accuracy: {round((1 - num_misclassified / len(y_test)) * 100, 2)}%") 

n_misclassified_to_show = min(num_misclassified, 5)  # Show up to 5 
misclassified images 
plt.figure(figsize=(10, 5)) 
for i in range(n_misclassified_to_show): 
idx = misclassified_idx[i] 
plt.subplot(1, n_misclassified_to_show, i + 1) 
plt.imshow(x_test[idx].reshape(64, 64), cmap='gray') 
plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}") 
plt.axis('off') 
plt.show()

  y_test_bin = label_binarize(y_test, classes=np.unique(y_test)) 

y_pred_prob = nb.predict_proba(x_test) 
 
for i in range(y_test_bin.shape[1]): 
roc_auc = roc_auc_score(y_test_bin[:, i], y_pred_prob[:, i]) 
print(f"Class {i} AUC: {roc_auc:.2f}")

  
