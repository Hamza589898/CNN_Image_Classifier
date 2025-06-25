# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load Dataset
(X_train,y_train),(X_test,y_test) = cifar10.load_data()

# Normalize Pixel Vlaues
X_train = X_train / 255
X_test = X_test / 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test,num_classes=10)

# Make Model
model = Sequential()

# Input Layer
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))

# Hidden Layer
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))

# Hidden Layer
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))

# Output Layer
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))

# Define Model
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile Model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Load Image
img = image.load_img('cat.jpg',target_size=(32,32))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array,axis=0)

# Define Classes For DL Model
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Improve Image Gernalization
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Train Model
model.fit(datagen.flow(X_train, y_train, batch_size=64),epochs=25, validation_data=(X_test, y_test))

# Check Accuracy
test_loss,test_acc = model.evaluate(X_test,y_test)
print(f"Check accuracy : {test_acc}")

# Predict Image
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction[0])
print("Predicted class:", classes[predicted_class])


y_test_label = np.argmax(y_test,axis=1)
y_per_label = np.argmax(model.predict(X_test), axis=1)

# Plot Heatmap For Confusion Matrics
cm = confusion_matrix(y_test_label, y_per_label)
plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True,fmt='d',xticklabels=classes,yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel("Actual")
plt.title("Confusion Matrics")
plt.show()

# Final Accuracy
acc = accuracy_score(y_test_label, y_per_label)
print(f"Accuracy : {acc}")
