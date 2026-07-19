import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("=" * 60)
print("TRAIN DYNAMIC GESTURE MODEL (LSTM)")
print("=" * 60)

# Load dataset
print("\nLoading dataset...")
with open('dynamic_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

X_train = dataset['X_train']
X_test = dataset['X_test']
y_train = dataset['y_train']
y_test = dataset['y_test']
label_names = dataset['label_names']

print(f"✓ Loaded")
print(f"  Training shape: {X_train.shape}")
print(f"  Testing shape: {X_test.shape}")
print(f"  Labels: {label_names}")

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# One-hot encoding
y_train_onehot = to_categorical(y_train_encoded)
y_test_onehot = to_categorical(y_test_encoded)

num_classes = len(label_names)

print(f"\nNumber of classes: {num_classes}")

# Build LSTM model
print("\n" + "=" * 60)
print("BUILDING MODEL")
print("=" * 60)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30, 126)),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'dynamic_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_test, y_test_onehot),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

loss, accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model and encoder
model.save('dynamic_model.h5')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\n✓ Model saved:")
print("  - dynamic_model.h5")
print("  - dynamic_model_best.h5")
print("  - label_encoder.pkl")

# Detailed evaluation
print("\n" + "=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

for i, label in enumerate(label_names):
    mask = y_test_encoded == i
    if np.sum(mask) > 0:
        class_acc = np.sum(y_pred_classes[mask] == i) / np.sum(mask)
        print(f"{label:12s}: {class_acc * 100:.2f}% ({np.sum(mask)} samples)")

print("\n" + "=" * 60)
print("NEXT STEP:")
print("python inference_dynamic.py")
print("=" * 60)
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("=" * 60)
print("TRAIN DYNAMIC GESTURE MODEL (LSTM)")
print("=" * 60)

# Load dataset
print("\nLoading dataset...")
with open('dynamic_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

X_train = dataset['X_train']
X_test = dataset['X_test']
y_train = dataset['y_train']
y_test = dataset['y_test']
label_names = dataset['label_names']

print(f"✓ Loaded")
print(f"  Training shape: {X_train.shape}")
print(f"  Testing shape: {X_test.shape}")
print(f"  Labels: {label_names}")

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# One-hot encoding
y_train_onehot = to_categorical(y_train_encoded)
y_test_onehot = to_categorical(y_test_encoded)

num_classes = len(label_names)

print(f"\nNumber of classes: {num_classes}")

# Build LSTM model
print("\n" + "=" * 60)
print("BUILDING MODEL")
print("=" * 60)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30, 126)),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'dynamic_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_test, y_test_onehot),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

loss, accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model and encoder
model.save('dynamic_model.h5')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\n✓ Model saved:")
print("  - dynamic_model.h5")
print("  - dynamic_model_best.h5")
print("  - label_encoder.pkl")

# Detailed evaluation
print("\n" + "=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

for i, label in enumerate(label_names):
    mask = y_test_encoded == i
    if np.sum(mask) > 0:
        class_acc = np.sum(y_pred_classes[mask] == i) / np.sum(mask)
        print(f"{label:12s}: {class_acc * 100:.2f}% ({np.sum(mask)} samples)")

print("\n" + "=" * 60)
print("NEXT STEP:")
print("python inference_dynamic.py")
print("=" * 60)