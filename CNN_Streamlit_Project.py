
# ==========================================================
# CNN IMAGE CLASSIFICATION AND STREAMLIT APPLICATION
# Dataset: MNIST
# ==========================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# ---------------- PART A: LOAD DATASET ----------------

print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# ---------------- IMAGE PREPROCESSING ----------------

# Normalize pixel values between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add the grayscale channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Training data after preprocessing:", x_train.shape)
print("Testing data after preprocessing:", x_test.shape)

# ---------------- CNN MODEL CREATION ----------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dense(10, activation="softmax")
])

# ---------------- COMPILE MODEL ----------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------- DISPLAY MODEL SUMMARY ----------------

model.summary()

print("CNN model created successfully!")
# ---------------- TRAIN THE CNN MODEL ----------------

print("\nStarting CNN training...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

print("\nCNN training completed!")

# ---------------- EVALUATE THE MODEL ----------------

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\nTest Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

# ---------------- VISUALIZE TRAINING RESULTS ----------------

# Plot training and validation accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_graph.png")
plt.show()


# Plot training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_graph.png")
plt.show()

print("Accuracy and loss graphs saved successfully!")



# ---------------- SAMPLE PREDICTIONS ----------------

# Predict digits from test images
predictions = model.predict(x_test[:20], verbose=0)

# Get the predicted digit with the highest probability
predicted_labels = np.argmax(predictions, axis=1)

# Display actual and predicted labels
plt.figure(figsize=(12, 8))

for i in range(20):
    plt.subplot(4, 5, i + 1)

    # Display the handwritten digit
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")

    actual = y_test[i]
    predicted = predicted_labels[i]

    # Mark correct and incorrect predictions
    if actual == predicted:
        color = "green"
    else:
        color = "red"

    plt.title(
        f"Actual: {actual}\nPredicted: {predicted}",
        color=color
    )

    plt.axis("off")

plt.suptitle("Actual vs Predicted Digits")
plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()

print("Sample predictions saved successfully!")

# Display incorrect predictions
incorrect_indices = np.where(
    predicted_labels != y_test[:20]
)[0]

print("Incorrect predictions among the first 20 images:")

if len(incorrect_indices) == 0:
    print("All first 20 predictions are correct.")
else:
    for i in incorrect_indices:
        print(
            f"Image {i}: Actual = {y_test[i]}, "
            f"Predicted = {predicted_labels[i]}"
        )

        
# ==========================================================
# PART B: STREAMLIT WEB APPLICATION
# ==========================================================

st.title("CNN Handwritten Digit Recognition")

st.write(
    "Upload an image of a handwritten digit from 0 to 9. "
    "The trained CNN model will predict the digit."
)

st.info(
    "For best results, upload a clear image containing "
    "one handwritten digit."
)

# Image upload section
uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("L")

    st.subheader("Uploaded Image")
    st.image(image, caption="Your uploaded digit")

    # Convert image to the format expected by the CNN
    image = image.resize((28, 28))

    image_array = np.array(image).astype("float32") / 255.0

    image_array = image_array.reshape(1, 28, 28, 1)

    # Predict the digit
    if st.button("Predict Digit"):

        probabilities = model.predict(
            image_array,
            verbose=0
        )

        predicted_digit = int(np.argmax(probabilities[0]))

        confidence = float(
            np.max(probabilities[0]) * 100
        )

        st.success(
            f"Predicted Digit: {predicted_digit}"
        )

        st.write(
            f"Prediction Confidence: {confidence:.2f}%"
        )

        st.progress(confidence / 100)

        st.caption(
            "Confidence represents the model's estimated "
            "probability for its selected digit, not a guarantee "
            "that the prediction is correct."
        )
