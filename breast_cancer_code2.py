# Import libraries
import os
import time
import cv2 as cv
from PIL import Image
import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import EarlyStopping # type: ignore
from keras.utils import normalize # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Conv2D, MaxPooling2D # type: ignore
from keras.layers import Activation, Flatten, Dense, Dropout # type: ignore


img_dir = "Dataset/images/"

benign_img = os.listdir(img_dir + "no/")
malignant_img = os.listdir(img_dir + "yes/")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
dataset = []
label = []

# Print the names of the images in the each directory

# print("Images in the no cancer directory:")
for x in range(len(benign_img)):
    # print(x,":", benign_img[x])
    break # remove this line and the one before to print all images names
print("Total number of no cancer images:",len(benign_img))
    
# print("\nImages in the cancer directory:")
for x in range(len(malignant_img)):
    # print(x,":", malignant_img[x])    
    break # remove this line and the one before to print all images names
print("Total number of cancer images:",len(malignant_img))

# Confirm images are in jpg format
for x in range(len(benign_img)):
    path_format = benign_img[x]
    if path_format.split(".")[-1] not in ALLOWED_EXTENSIONS:
        print("Image format is not correct")
print("Finished checking no cancer images")
        
for x in range(len(malignant_img)):
    path_format = malignant_img[x]
    if path_format.split(".")[-1] not in ALLOWED_EXTENSIONS:
        print("Image format is not correct")
print("Finished checking cancer images")


# Read images
for x, img in enumerate(benign_img):
    
    start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    if path_format.split(".")[-1] in ALLOWED_EXTENSIONS:
        image = cv.imread(img_dir + "no/" + img)
        # cv.imshow("Image {}".format(img), image)
        image = Image.fromarray(image, "RGB")
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)
        
        end_time = time.time()
        print(f"Processed {img}{timestamp}")
        print(f"Time taken to process {img}: {end_time - start_time:.2f} seconds")
        
        
# Read images
for x, img in enumerate(malignant_img):
        
    start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    if path_format.split(".")[-1] in ALLOWED_EXTENSIONS:
        image = cv.imread(img_dir + "yes/" + img)
        # cv.imshow("Image {}".format(img), image)
        import time
        image = Image.fromarray(image, "RGB")
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)
        
        end_time = time.time()
        print(f"Processed {img} at {timestamp}")
        print(f"Time taken to process {img}: {end_time - start_time:.2f} seconds")

print("\nFinished processing all images.\n")

# Print the length of the dataset and labels
# print(dataset, len(dataset))
# print(label, len(label))

# Convert dataset and labels to numpy arrays
dataset = np.array(dataset)
label = np.array(label)

# Print the shape of the dataset and labels
# print("Shape of dataset:", dataset.shape)
# print("Shape of labels:", label.shape)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

# print("Shape of training set:", X_train.shape)
# print("Shape of training set:", X_test.shape)
# print("Shape of test set:", y_test.shape)
# print("Shape of testing set:", y_test.shape)

# Normalize the data
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)
print("Shape of training set after normalization:", X_train.shape)
print("Shape of testing set after normalization:", X_test.shape)


# Build model
model = Sequential()
#input_size = (64, 64) for input_shape
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add more convolutional layers
model.add(Conv2D(32, (3, 3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add more convolutional layers
model.add(Conv2D(64, (3, 3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add more convolutional layers
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))


# Compile and save the model
print("\nCompiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print("\nTraining model...")
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
model.fit(X_train, y_train, batch_size=16, verbose=1,
          epochs=30, validation_data = (X_test, y_test),
          callbacks=[early_stop], shuffle = False)

print(model.summary())

print("\nSaving model...")
model.save("Models_created/breast_cancer_model2.h5")
print("Model saved successfully!")

# Epoch 6/30
# accuracy: 0.6790 - loss: 0.6131 - val_accuracy: 0.7011 - val_loss: 0.6183