import dlib
import cv2
import numpy as np
import os
import json
import time
from datetime import datetime

# === Paths ===
PREDICTOR_PATH = r"C:\Users\divya\DDD_CNN_PE\shape_predictor_68_face_landmarks.dat"
TRAIN_LIST_PATH = r"C:\Users\divya\DDD_CNN_PE\train.txt"
TEST_LIST_PATH = r"C:\Users\divya\DDD_CNN_PE\test.txt"
PREPROCESSED_DIR = r"C:\Users\divya\DDD_CNN_PE\preprocessed_data"

# Create preprocessed data directory
if not os.path.exists(PREPROCESSED_DIR):
    os.makedirs(PREPROCESSED_DIR)

# === Initializations ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def print_banner(title):
    """Print a beautiful banner"""
    banner_width = 80
    print("\n" + "="*banner_width)
    print(f"{'':^{banner_width}}")
    print(f"{title.upper()}".center(banner_width))
    print(f"{'':^{banner_width}}")
    print("="*banner_width)

def print_progress_bar(current, total, bar_length=50, prefix="Progress"):
    """Print a visual progress bar"""
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\r{prefix}: [{arrow + spaces}] {percent:.1f}% ({current}/{total})', end='', flush=True)

def extract_landmarks(image_path):
    """Extracts 68 facial landmarks (x,y) from a grayscale image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, "Failed to load image"
    
    faces = detector(image, 1)
    if len(faces) == 0:
        return None, "No face detected"
    
    landmarks = predictor(image, faces[0])
    coords = []
    for i in range(68):
        coords.append(landmarks.part(i).x)
        coords.append(landmarks.part(i).y)
    
    return np.array(coords, dtype='float32'), "Success"

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
    """Loads image paths and labels from a txt file."""
    image_paths, labels = [], []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                image_path = parts[0]
                if parts[1].endswith('.txt'):  # label file path
                    try:
                        with open(parts[1], 'r') as lf:
                            label_content = lf.readline().strip()
                            if label_content:
                                label = int(label_content.split()[0])
                            else:
                                continue
                    except:
                        continue
                else:  # direct label
                    label = int(parts[1])
                image_paths.append(image_path)
                labels.append(label)
    return image_paths, labels

def preprocess_and_save_dataset(txt_path, output_prefix):
    """
    Preprocesses dataset and saves to compressed numpy files.
    
    Args:
        txt_path: Path to train.txt or test.txt
        output_prefix: Prefix for output files (e.g., 'train' or 'test')
    
    Returns:
        Dictionary with processing statistics
    """
    print_banner(f"Processing {output_prefix.upper()} Dataset")
    
    # Load image paths and labels
    print(f"📂 Loading data from: {os.path.basename(txt_path)}")
    image_paths, labels = load_data_from_txt(txt_path)
    print(f"✅ Found {len(image_paths)} samples to process")
    
    # Initialize arrays and counters
    X, y = [], []
    successful = 0
    failed = 0
    error_log = []
    
    print(f"\n🔍 Extracting facial landmarks...")
    start_time = time.time()
    
    # Process each image
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        print_progress_bar(i + 1, len(image_paths), prefix="Processing")
        
        landmarks, status = extract_landmarks(image_path)
        
        if landmarks is not None and len(landmarks) == 136:
            # Normalize landmarks
            normalized_landmarks = normalize_landmarks(landmarks)
            X.append(normalized_landmarks)
            y.append(label)
            successful += 1
        else:
            failed += 1
            error_log.append({
                'image_path': image_path,
                'label': label,
                'error': status,
                'index': i
            })
    
    processing_time = time.time() - start_time
    print(f"\n✅ Processing completed in {processing_time:.2f} seconds")
    
    # Convert to numpy arrays
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')
    
    # Save processed data
    output_data_path = os.path.join(PREPROCESSED_DIR, f"{output_prefix}_data.npz")
    output_labels_path = os.path.join(PREPROCESSED_DIR, f"{output_prefix}_labels.npy")
    output_stats_path = os.path.join(PREPROCESSED_DIR, f"{output_prefix}_stats.json")
    output_errors_path = os.path.join(PREPROCESSED_DIR, f"{output_prefix}_errors.json")
    
    print(f"\n💾 Saving preprocessed data...")
    
    # Save data and labels
    np.savez_compressed(output_data_path, features=X, labels=y)
    np.save(output_labels_path, y)
    
    # Create statistics
    stats = {
        'dataset': output_prefix,
        'total_samples': len(image_paths),
        'successful_samples': successful,
        'failed_samples': failed,
        'success_rate': successful / len(image_paths) * 100,
        'feature_shape': X.shape,
        'label_shape': y.shape,
        'processing_time_seconds': processing_time,
        'unique_labels': np.unique(y).tolist(),
        'label_distribution': {int(label): int(count) for label, count in zip(*np.unique(y, return_counts=True))},
        'saved_files': {
            'data': output_data_path,
            'labels': output_labels_path,
            'stats': output_stats_path,
            'errors': output_errors_path
        },
        'processed_at': datetime.now().isoformat()
    }
    
    # Save statistics
    with open(output_stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Save error log
    if error_log:
        with open(output_errors_path, 'w') as f:
            json.dump(error_log, f, indent=4)
    
    # Print summary
    summary_info = f"""
📊 PROCESSING SUMMARY FOR {output_prefix.upper()}:
   ✅ Successful samples: {successful}
   ❌ Failed samples: {failed}
   📈 Success rate: {stats['success_rate']:.2f}%
   📐 Feature shape: {X.shape}
   🏷️  Label shape: {y.shape}
   ⏱️  Processing time: {processing_time:.2f} seconds
   📁 Data saved to: {output_data_path}
   📋 Stats saved to: {output_stats_path}
"""
    print(summary_info)
    
    if error_log:
        print(f"   ⚠️  Error log saved to: {output_errors_path}")
    
    return stats

def load_preprocessed_data(dataset_type):
    """
    Load preprocessed data from saved files.
    
    Args:
        dataset_type: 'train' or 'test'
    
    Returns:
        Tuple of (X, y) numpy arrays
    """
    data_path = os.path.join(PREPROCESSED_DIR, f"{dataset_type}_data.npz")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Preprocessed {dataset_type} data not found at {data_path}")
    
    data = np.load(data_path)
    return data['features'], data['labels']

def verify_preprocessed_data():
    """Verify the integrity of preprocessed data files."""
    print_banner("Verifying Preprocessed Data")
    
    datasets = ['train', 'test']
    verification_results = {}
    
    for dataset in datasets:
        try:
            # Load data
            X, y = load_preprocessed_data(dataset)
            
            # Load stats
            stats_path = os.path.join(PREPROCESSED_DIR, f"{dataset}_stats.json")
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            # Verify data integrity
            verification = {
                'dataset': dataset,
                'data_loaded': True,
                'feature_shape_matches': list(X.shape) == stats['feature_shape'],
                'label_shape_matches': list(y.shape) == stats['label_shape'],
                'sample_count_matches': len(X) == stats['successful_samples'],
                'feature_range': [float(X.min()), float(X.max())],
                'unique_labels': np.unique(y).tolist()
            }
            
            verification_results[dataset] = verification
            
            status = "✅" if all([verification['data_loaded'], verification['feature_shape_matches'], 
                                verification['label_shape_matches'], verification['sample_count_matches']]) else "❌"
            
            print(f"{status} {dataset.upper()} Dataset:")
            print(f"   📐 Features: {X.shape}")
            print(f"   🏷️  Labels: {y.shape}")
            print(f"   📊 Feature range: [{verification['feature_range'][0]:.4f}, {verification['feature_range'][1]:.4f}]")
            print(f"   🎯 Unique labels: {verification['unique_labels']}")
            
        except Exception as e:
            print(f"❌ {dataset.upper()} Dataset: Error - {str(e)}")
            verification_results[dataset] = {'dataset': dataset, 'error': str(e)}
    
    return verification_results

def main():
    """Main preprocessing pipeline."""
    print("🚀 DATASET PREPROCESSING PIPELINE")
    print("="*80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Preprocessed data will be saved to: {PREPROCESSED_DIR}")
    
    # Check if predictor file exists
    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(f"❌ Shape predictor file not found: {PREDICTOR_PATH}")
    
    # Process training data
    if os.path.exists(TRAIN_LIST_PATH):
        train_stats = preprocess_and_save_dataset(TRAIN_LIST_PATH, 'train')
    else:
        print(f"⚠️  Training data file not found: {TRAIN_LIST_PATH}")
        train_stats = None
    
    # Process testing data
    if os.path.exists(TEST_LIST_PATH):
        test_stats = preprocess_and_save_dataset(TEST_LIST_PATH, 'test')
    else:
        print(f"⚠️  Testing data file not found: {TEST_LIST_PATH}")
        test_stats = None
    
    # Verify preprocessed data
    verification_results = verify_preprocessed_data()
    
    # Final summary
    print_banner("Final Summary")
    total_successful = 0
    total_samples = 0
    
    if train_stats:
        total_successful += train_stats['successful_samples']
        total_samples += train_stats['total_samples']
        print(f"📈 Training: {train_stats['successful_samples']}/{train_stats['total_samples']} samples ({train_stats['success_rate']:.2f}%)")
    
    if test_stats:
        total_successful += test_stats['successful_samples']
        total_samples += test_stats['total_samples']
        print(f"📈 Testing: {test_stats['successful_samples']}/{test_stats['total_samples']} samples ({test_stats['success_rate']:.2f}%)")
    
    overall_success_rate = total_successful / total_samples * 100 if total_samples > 0 else 0
    print(f"🎯 Overall: {total_successful}/{total_samples} samples ({overall_success_rate:.2f}%)")
    print(f"💾 All preprocessed data saved to: {PREPROCESSED_DIR}")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save final summary
    final_summary = {
        'preprocessing_completed_at': datetime.now().isoformat(),
        'train_stats': train_stats,
        'test_stats': test_stats,
        'verification_results': verification_results,
        'overall_stats': {
            'total_samples': total_samples,
            'total_successful': total_successful,
            'overall_success_rate': overall_success_rate
        }
    }
    
    summary_path = os.path.join(PREPROCESSED_DIR, 'preprocessing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=4)
    
    print(f"📋 Final summary saved to: {summary_path}")

if __name__ == "__main__":
    main()