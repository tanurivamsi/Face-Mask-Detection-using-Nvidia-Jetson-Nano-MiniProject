import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

data = []
labels = []

categories = ["with_mask", "without_mask"]

base_path = r"C:\Users\tanur\OneDrive\Desktop\final\dataset"

for category in categories:
    path = os.path.join(base_path, category)  # Now correctly points to each subfolder
    label = categories.index(category)

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (100, 100))
        data.append(img)
        labels.append(label)

# Check if data is loaded
print(f"Total images loaded: {len(data)}")

# Convert to NumPy arrays
data = np.array(data) / 255.0
labels = to_categorical(np.array(labels), 2)

# Make sure we have enough data
if len(data) == 0:
    print("No images found. Check folder paths.")
else:
    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(f"Training samples: {len(x_train)}, Testing samples: {len(x_test)}")
    


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)




model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(aug.flow(x_train, y_train, batch_size=32),
          validation_data=(x_test, y_test),
          steps_per_epoch=len(x_train) // 32,
          epochs=10)

model.save("mask_detector_model.h5")



model = load_model("mask_detector_model.h5")
categories = ["Mask", "No Mask"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face)[0]
        label = categories[np.argmax(pred)]
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
