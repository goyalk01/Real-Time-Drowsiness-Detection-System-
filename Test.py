import dlib
import cv2
import numpy as np
import os
from keras.layers import BatchNormalization, ELU
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LeakyReLU
from keras.utils import to_categorical

# === Paths (adjust these as needed) ===
PREDICTOR_PATH = r"C:\Users\divya\DDD_CNN_PE\shape_predictor_68_face_landmarks.dat"
TRAIN_LIST_PATH = r"C:\Users\divya\DDD_CNN_PE\train.txt"
TEST_LIST_PATH = r"C:\Users\divya\DDD_CNN_PE\test.txt"
MODEL_SAVE_NAME = r"C:\Users\divya\DDD_CNN_PE\drowsiness_cnn_model.h5"

# === Initializations ===
detector = dlib.get_frontal_face_detector()  # use HOG detector for speed and portability
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def extract_landmarks(image_path, show=True):
    """Extracts 68 facial landmarks (x,y) from a grayscale image and shows progress."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    basename = os.path.basename(image_path)
    if image is None:
        print(f"Failed to load image: {os.path.abspath(image_path)}")
        return None
    if show:
        # Show image with overlayed status text
        image_disp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.putText(image_disp, f"Processing: {basename}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Processing Image', image_disp)
        # Wait 1ms and allow for "q" to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Aborting processing early (user quit).")
            exit(0)

    faces = detector(image, 1)
    if len(faces) == 0:
        print(f"No face detected: {os.path.abspath(image_path)}")
        if show:
            image_disp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.putText(image_disp, f"No face: {basename}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Processing Image', image_disp)
            cv2.waitKey(500)  # show for 0.5 sec or adjust as needed
        return None
    landmarks = predictor(image, faces[0])
    coords = []
    for i in range(68):
        coords.append(landmarks.part(i).x)
        coords.append(landmarks.part(i).y)
    # Draw landmarks for visual confirmation
    if show:
        image_disp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i in range(68):
            pt = (landmarks.part(i).x, landmarks.part(i).y)
            cv2.circle(image_disp, pt, 2, (255, 0, 0), -1)
        cv2.putText(image_disp, f"Landmarks: {basename}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Processing Image', image_disp)
        cv2.waitKey(10)  # brief pause so you can see it
    return np.array(coords, dtype='float32')

def normalize_landmarks(landmarks):
    """Min-max normalizes landmarks per face."""
    x = landmarks[0::2]
    y = landmarks[1::2]
    x_norm = (x - np.min(x)) / (np.ptp(x) + 1e-6)
    y_norm = (y - np.min(y)) / (np.ptp(y) + 1e-6)
    norm = np.empty_like(landmarks)
    norm[0::2] = x_norm
    norm[1::2] = y_norm
    return norm

def load_data_from_txt(txt_path):
    """Loads image paths and labels from a txt file (format: path label)."""
    image_paths, labels = [], []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label = line.rsplit(' ', 1)
            image_paths.append(path)
            labels.append(int(label))
    return image_paths, labels

def prepare_data(txt_path, show=True):
    """Loads, extracts, and normalizes all data and returns (X, y) numpy arrays with display."""
    image_paths, labels = load_data_from_txt(txt_path)
    X, y = [], []
    for i, (path, label) in enumerate(zip(image_paths, labels)):
        print(f"[{i+1}/{len(image_paths)}] Processing {path} (label: {label})")
        landmarks = extract_landmarks(path, show=show)
        if landmarks is not None and len(landmarks) == 136:
            landmarks = normalize_landmarks(landmarks)
            X.append(landmarks)
            y.append(label)
        else:
            print(f"Skipping: {os.path.abspath(path)} (index {i})")
    cv2.destroyAllWindows()  # Close the image display window after processing
    return np.array(X), np.array(y)

def create_drowsiness_detection_model():
    """Creates an optimized CNN model for 136-point landmark features."""
    model = Sequential()
    model.add(Conv1D(64, 3, padding='same', input_shape=(136,1), kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.3))

    model.add(Conv1D(128, 3, padding='same', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.3))

    model.add(Conv1D(256, 3, padding='same', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.4))

    model.add(Conv1D(512, 3, padding='same', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
def main():
    print("Loading and preprocessing training data...")
    X_train, y_train = prepare_data(TRAIN_LIST_PATH, show=True)
    print(f"Training samples processed: {len(X_train)}")
    print("Loading and preprocessing testing data...")
    X_test, y_test = prepare_data(TEST_LIST_PATH, show=True)
    print(f"Testing samples processed: {len(X_test)}")
    # Safeguard: Ensure non-empty datasets
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("No training data found. Check landmark extraction and face detection logic.")
    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("No test data found. Check landmark extraction and face detection logic.")
    # Reshape for Conv1D: (samples, 136, 1)
    X_train = X_train.reshape(-1, 136, 1)
    X_test = X_test.reshape(-1, 136, 1)
    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)
    print("Creating model...")
    model = create_drowsiness_detection_model()
    print("Training model...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=30,
        batch_size=128,
        shuffle=True
    )
    print("Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    model.save(MODEL_SAVE_NAME)
    print(f"Model saved as {MODEL_SAVE_NAME}")

if __name__ == "__main__":
    main()
