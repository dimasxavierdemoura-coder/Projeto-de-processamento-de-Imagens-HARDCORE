import glob
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from image_processing import extract_features_from_path, load_image


def collect_image_paths(dataset_dir, extensions=None):
    if extensions is None:
        extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().split('.')[-1] in extensions:
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)


def load_dataset_features(dataset_dir, extensions=None, balance=False):
    image_paths = collect_image_paths(dataset_dir, extensions=extensions)
    records = []
    labels = []
    for path in image_paths:
        try:
            label = os.path.basename(os.path.dirname(path))
            records.append(extract_features_from_path(path))
            labels.append(label)
        except Exception as e:
            print(f"Aviso: pulando arquivo inválido {path}: {e}")
    df = pd.DataFrame(records)
    X = df.fillna(0).values
    y = np.array(labels)
    
    if balance:
        print("Aplicando balanceamento com RandomOverSampler...")
        ros = RandomOverSampler(random_state=42)
        X_balanced, y_balanced = ros.fit_resample(X, y)
        print(f"Dataset original: {len(X)} amostras")
        print(f"Dataset balanceado: {len(X_balanced)} amostras")
        return X_balanced, y_balanced
    
    return X, y


def load_dataset_images(dataset_dir, target_size=(128, 128), extensions=None, balance=False, augment=False):
    """Load images directly for CNN 2D training"""
    image_paths = collect_image_paths(dataset_dir, extensions=extensions)
    images = []
    labels = []
    
    for path in image_paths:
        try:
            label = os.path.basename(os.path.dirname(path))
            img = load_image(path)
            
            # Resize image
            import cv2
            img_resized = cv2.resize(img, target_size)
            
            # Normalize to [0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            images.append(img_normalized)
            labels.append(label)
        except Exception as e:
            print(f"Aviso: pulando arquivo inválido {path}: {e}")
    
    X = np.array(images)
    y = np.array(labels)
    
    if balance:
        print("Aplicando balanceamento com RandomOverSampler...")
        # For images, we need to reshape for oversampling
        X_reshaped = X.reshape(X.shape[0], -1)
        ros = RandomOverSampler(random_state=42)
        X_balanced_reshaped, y_balanced = ros.fit_resample(X_reshaped, y)
        X_balanced = X_balanced_reshaped.reshape(-1, target_size[0], target_size[1], 3)
        print(f"Dataset original: {len(X)} amostras")
        print(f"Dataset balanceado: {len(X_balanced)} amostras")
        return X_balanced, y_balanced
    
    if augment:
        print("Aplicando data augmentation...")
        X_aug, y_aug = augment_images(X, y)
        print(f"Dataset original: {len(X)} amostras")
        print(f"Dataset aumentado: {len(X_aug)} amostras")
        return X_aug, y_aug
    
    return X, y


def augment_images(X, y, augment_factor=3):
    """Apply data augmentation to increase dataset size"""
    augmented_images = []
    augmented_labels = []
    
    for img, label in zip(X, y):
        augmented_images.append(img)
        augmented_labels.append(label)
        
        # Apply augmentations
        for _ in range(augment_factor):
            # Random rotation
            angle = np.random.uniform(-30, 30)
            import cv2
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            rotated = cv2.warpAffine(img, M, (w, h))
            augmented_images.append(rotated)
            augmented_labels.append(label)
            
            # Random flip
            if np.random.random() > 0.5:
                flipped = cv2.flip(img, 1)  # Horizontal flip
                augmented_images.append(flipped)
                augmented_labels.append(label)
            
            # Random zoom
            if np.random.random() > 0.7:
                zoom_factor = np.random.uniform(0.8, 1.2)
                h, w = img.shape[:2]
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                zoomed = cv2.resize(img, (new_w, new_h))
                # Crop or pad to original size
                if zoom_factor > 1:
                    # Crop center
                    start_h = (new_h - h) // 2
                    start_w = (new_w - w) // 2
                    zoomed = zoomed[start_h:start_h+h, start_w:start_w+w]
                else:
                    # Pad with zeros
                    pad_h = (h - new_h) // 2
                    pad_w = (w - new_w) // 2
                    zoomed = cv2.copyMakeBorder(zoomed, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, cv2.BORDER_CONSTANT, value=0)
                augmented_images.append(zoomed)
                augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)


def grid_search_cnn(X, y, param_grid, cv=3):
    """Perform grid search for CNN hyperparameters"""
    from sklearn.model_selection import StratifiedKFold
    import tensorflow as tf
    
    best_score = 0
    best_params = None
    best_model = None
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    for params in param_grid:
        scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Encode labels
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train_fold)
            y_val_encoded = le.transform(y_val_fold)
            
            # Build model with current params
            input_shape = X_train_fold.shape[1:]
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Conv2D(params['filters1'], (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(params['filters2'], (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(params['dense_units'], activation='relu'),
                tf.keras.layers.Dropout(params['dropout_rate']),
                tf.keras.layers.Dense(len(np.unique(y_train_encoded)), activation='softmax')
            ])
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
            model.compile(optimizer=optimizer,
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
            
            # Quick training for grid search
            model.fit(X_train_fold, y_train_encoded, epochs=5, batch_size=32, verbose=0)
            
            # Evaluate
            val_loss, val_acc = model.evaluate(X_val_fold, y_val_encoded, verbose=0)
            scores.append(val_acc)
        
        avg_score = np.mean(scores)
        print(f"Params: {params}, Avg CV Score: {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            # Train final model with best params
            best_model, _ = train_classifier(X, y, model_type="cnn_2d", **params)
    
    print(f"Best params: {best_params}, Best CV score: {best_score:.4f}")
    return best_model, best_params, best_score


def benchmark_models(dataset_dir, test_size=0.2):
    """Benchmark RandomForest vs CNN on the same dataset"""
    print("=== BENCHMARK: RandomForest vs CNN 2D ===")
    
    # Load images for CNN
    print("Loading images for CNN...")
    X_images, y_images = load_dataset_images(dataset_dir, target_size=(128, 128), balance=True)
    
    # Load features for RF
    print("Loading features for RandomForest...")
    X_features, y_features = load_dataset_features(dataset_dir, balance=True)
    
    results = {}
    
    # Train RandomForest
    print("\n--- Training RandomForest ---")
    rf_model, rf_report = train_classifier(X_features, y_features, model_type="rf")
    results['rf'] = {
        'report': rf_report,
        'training_time': None,  # Could add timing
        'data_type': 'features'
    }
    print(f"RandomForest - Accuracy: {rf_report['accuracy']:.4f}")
    print(f"RandomForest - Macro F1: {rf_report['macro avg']['f1-score']:.4f}")
    print(f"RandomForest - Weighted F1: {rf_report['weighted avg']['f1-score']:.4f}")
    
    # Train CNN 2D
    print("\n--- Training CNN 2D ---")
    cnn_model, cnn_report = train_classifier(X_images, y_images, model_type="cnn_2d", epochs=15)
    results['cnn_2d'] = {
        'report': cnn_report,
        'training_time': None,
        'data_type': 'images'
    }
    print(f"CNN 2D - Accuracy: {cnn_report['accuracy']:.4f}")
    print(f"CNN 2D - Macro F1: {cnn_report['macro avg']['f1-score']:.4f}")
    print(f"CNN 2D - Weighted F1: {cnn_report['weighted avg']['f1-score']:.4f}")
    
    # Compare results
    print("\n=== COMPARISON ===")
    print(f"RandomForest Accuracy: {results['rf']['report']['accuracy']:.4f}")
    print(f"CNN 2D Accuracy: {results['cnn_2d']['report']['accuracy']:.4f}")
    
    winner = 'rf' if results['rf']['report']['accuracy'] > results['cnn_2d']['report']['accuracy'] else 'cnn_2d'
    print(f"Winner: {winner.upper()}")
    
    return results


def train_classifier(X, y, test_size=0.2, random_state=42, model_type="rf", **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
    elif model_type == "cnn_1d":
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Encode labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        # Reshape for CNN (assuming features are flattened)
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1, 1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1, 1)
        
        # Build CNN model
        model = keras.Sequential([
            layers.Input(shape=(X_train_reshaped.shape[1], 1)),
            layers.Conv1D(32, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(np.unique(y_train_encoded)), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Train CNN
        model.fit(X_train_reshaped, y_train_encoded, epochs=10, batch_size=32, 
                 validation_split=0.2, verbose=1)
        
        # Predictions
        predictions_proba = model.predict(X_test_reshaped)
        predictions_encoded = np.argmax(predictions_proba, axis=1)
        predictions = le.inverse_transform(predictions_encoded)
        
        # Store label encoder for prediction
        model.label_encoder = le
        
    elif model_type == "cnn_2d":
        import tensorflow as tf
        from tensorflow.keras import layers
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Encode labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        # Build CNN 2D model
        input_shape = X_train.shape[1:]  # (height, width, channels)
        
        # Get hyperparameters from kwargs
        filters1 = kwargs.get('filters1', 32)
        filters2 = kwargs.get('filters2', 64)
        dense_units = kwargs.get('dense_units', 128)
        dropout_rate = kwargs.get('dropout_rate', 0.5)
        learning_rate = kwargs.get('learning_rate', 0.001)
        
        model = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(filters1, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(filters2, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(dense_units, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(len(np.unique(y_train_encoded)), activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Data augmentation during training
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Train CNN
        epochs = kwargs.get('epochs', 20)
        batch_size = kwargs.get('batch_size', 32)
        
        model.fit(datagen.flow(X_train, y_train_encoded, batch_size=batch_size),
                 epochs=epochs,
                 validation_data=(X_test, y_test_encoded),
                 verbose=1)
        
        # Predictions
        predictions_proba = model.predict(X_test)
        predictions_encoded = np.argmax(predictions_proba, axis=1)
        predictions = le.inverse_transform(predictions_encoded)
        
        # Store label encoder for prediction
        model.label_encoder = le
    
    report = classification_report(y_test, predictions, output_dict=True)
    report["accuracy"] = accuracy_score(y_test, predictions)
    return model, report


def grid_search_cnn(X_images, y_images, param_grid=None):
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32],
            'epochs': [10, 20]
        }
    
    from sklearn.model_selection import ParameterGrid
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import accuracy_score
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_images)
    y_cat = to_categorical(y_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_images, y_cat, test_size=0.2, random_state=42, stratify=y_encoded)
    
    best_score = 0
    best_params = None
    
    for params in ParameterGrid(param_grid):
        print(f"Testando parâmetros: {params}")
        
        # Build model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(np.unique(y_encoded)), activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Train
        model.fit(X_train, y_train, 
                 epochs=params['epochs'], 
                 batch_size=params['batch_size'],
                 verbose=0)
        
        # Evaluate
        predictions_proba = model.predict(X_test, verbose=0)
        predictions_encoded = np.argmax(predictions_proba, axis=1)
        y_test_decoded = np.argmax(y_test, axis=1)
        score = accuracy_score(y_test_decoded, predictions_encoded)
        
        print(".4f")
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score


def save_model(model, path):
    # Check if it's a CNN model
    if hasattr(model, 'label_encoder'):
        # Save CNN model and label encoder separately
        model_path = path.replace('.joblib', '_model.h5')
        encoder_path = path.replace('.joblib', '_encoder.joblib')
        model.save(model_path)
        joblib.dump(model.label_encoder, encoder_path)
        # Save metadata
        metadata = {'type': 'cnn', 'model_path': model_path, 'encoder_path': encoder_path}
        joblib.dump(metadata, path)
    else:
        joblib.dump(model, path)


def load_model(path):
    metadata = joblib.load(path)
    if isinstance(metadata, dict) and metadata.get('type') == 'cnn':
        # Load CNN model
        import tensorflow as tf
        model = tf.keras.models.load_model(metadata['model_path'])
        model.label_encoder = joblib.load(metadata['encoder_path'])
        return model
    else:
        # Load traditional model
        return metadata


def predict_image(model, image_path):
    # Check if it's a CNN model (has label_encoder attribute)
    if hasattr(model, 'label_encoder'):
        import cv2
        # Load and preprocess image
        img = load_image(image_path)
        img_resized = cv2.resize(img, (128, 128))  # Same size as training
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Check if CNN 2D (4D input) or CNN 1D (3D input)
        if len(model.input_shape) == 4:  # CNN 2D: (batch, height, width, channels)
            img_reshaped = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
            prediction_proba = model.predict(img_reshaped)
        else:  # CNN 1D: features-based
            features = extract_features_from_path(image_path)
            features_array = np.array([features])
            features_reshaped = features_array.reshape(features_array.shape[0], -1, 1)
            prediction_proba = model.predict(features_reshaped)
        
        prediction_encoded = np.argmax(prediction_proba, axis=1)
        prediction = model.label_encoder.inverse_transform(prediction_encoded)
        return prediction[0]
    else:
        # RandomForest prediction
        features = extract_features_from_path(image_path)
        df = pd.DataFrame([features])
        prediction = model.predict(df)
        return prediction[0]
