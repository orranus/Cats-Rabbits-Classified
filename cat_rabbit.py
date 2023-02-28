import os
import numpy as np
from PIL import Image
from sklearn.naive_bayes import GaussianNB

# Define image directories
kitty_train = '/Users/orranus/Documents/CS377/Dataset/Cat-Rabbit/train-cat-rabbit/cat'
rabbit_train = '/Users/orranus/Documents/CS377/Dataset/Cat-Rabbit/train-cat-rabbit/rabbit'
test_folder = '/Users/orranus/Documents/CS377/Dataset/Cat-Rabbit/test-images'

# Read images and create image matrix
X = []
y = []

# Read Braeburn apple images
for img_file in os.listdir(kitty_train):
    if img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png'):  # Skip non-image files
        img = Image.open(os.path.join(kitty_train, img_file)).convert('L').resize((8,8))
        X.append(np.array(img).flatten())
        y.append('Cat')

# Read Crimson Snow apple images
for img_file in os.listdir(rabbit_train):
    if img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png'):  # Skip non-image files
        img = Image.open(os.path.join(rabbit_train, img_file)).convert('L').resize((8,8))
        X.append(np.array(img).flatten())
        y.append('Rabbit')

# Train Naive Bayes classifier
clf = GaussianNB()
clf.fit(X, y)

# Predict the class of new images in the test_folder
for img_file in os.listdir(test_folder):
    if img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png'):  # Skip non-image files
        test_image = Image.open(os.path.join(test_folder, img_file)).convert('L').resize((8,8))
        new_X = np.array(test_image).flatten()
        testimg = Image.open(os.path.join(test_folder, img_file))

        # Predict the class of the new image
        predicted = clf.predict([new_X])
        print('Image:', img_file, 'Predicted class:', predicted[0])