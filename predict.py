import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import os

print("Cattle Disease Prediction System")
print("-" * 40)

# Load model and label encoder
print("Loading model...")
model = tf.keras.models.load_model("best_cattle_disease_model.h5")
label_encoder = joblib.load("label_encoder.pkl")
print("Model loaded successfully!")
print("Classes:", list(label_encoder.classes_))

def predict(image_path):
    """Predict disease from a cattle image."""

    # Load and resize image
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    idx        = np.argmax(predictions[0])
    disease    = label_encoder.inverse_transform([idx])[0]
    confidence = predictions[0][idx]

    # Print result
    print(f"\nImage     : {os.path.basename(image_path)}")
    print(f"Prediction: {disease}")
    print(f"Confidence: {confidence*100:.1f}%")

    # Advice
    advice = {
        "Bovine Mastitis":  "Please consult a vet. Udder infection detected.",
        "Dermatophilosis":  "Keep animal dry. Consult vet for antibiotics.",
        "Healthy":          "Animal appears healthy.",
        "Pediculosis_aug":  "Lice detected. Apply insecticide treatment.",
        "Ringworm_aug":     "Fungal infection. Apply antifungal cream.",
        "lumpy skin":       "URGENT: Isolate animal immediately. Viral disease.",
        "pinkeye":          "Eye infection. Apply antibiotic eye drops.",
    }
    print("Advice    :", advice.get(disease, "Consult a veterinarian."))

    # Show chart
    all_diseases = label_encoder.classes_
    all_probs    = predictions[0]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(image_path))
    plt.title(f"Prediction: {disease}\nConfidence: {confidence*100:.1f}%")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    colors = ["green" if d == disease else "steelblue" for d in all_diseases]
    plt.barh(all_diseases, all_probs, color=colors)
    plt.xlabel("Probability")
    plt.title("Disease Probabilities")
    plt.tight_layout()
    plt.show()

    return disease, confidence


# Run prediction
print("\nEnter path to a cattle image (or press Enter to skip):")
image_path = input("Image path: ").strip()

if image_path and os.path.exists(image_path):
    predict(image_path)
elif image_path:
    print("File not found:", image_path)
else:
    print("No image entered. Run predict('your_image.jpg') to test.")