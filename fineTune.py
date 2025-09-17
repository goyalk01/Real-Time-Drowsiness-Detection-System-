import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import BatchNormalization, ELU
from keras.models import Sequential, load_model
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# === Paths ===
PREPROCESSED_DIR = r"C:\Users\divya\DDD_CNN_PE\preprocessed_data"
MODEL_SAVE_NAME = r"C:\Users\divya\DDD_CNN_PE\drowsiness_cnn_final_model.h5"
CHECKPOINT_DIR = r"C:\Users\divya\DDD_CNN_PE\phase_checkpoints"

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

def print_phase_banner(phase_num, phase_name):
    """Print a beautiful banner for each phase"""
    banner_width = 80
    print("\n" + "="*banner_width)
    print(f"{'':^{banner_width}}")
    print(f"PHASE {phase_num}: {phase_name.upper()}".center(banner_width))
    print(f"{'':^{banner_width}}")
    print("="*banner_width)

def load_preprocessed_data():
    """Load preprocessed data from saved .npz files"""
    print("\n📂 Loading preprocessed data...")
    
    # Load training data
    train_path = os.path.join(PREPROCESSED_DIR, "train_data.npz")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ Training data not found at: {train_path}")
    
    train_data = np.load(train_path)
    X_train, y_train = train_data['features'], train_data['labels']
    
    # Load testing data
    test_path = os.path.join(PREPROCESSED_DIR, "test_data.npz")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ Testing data not found at: {test_path}")
    
    test_data = np.load(test_path)
    X_test, y_test = test_data['features'], test_data['labels']
    
    print(f"✅ Successfully loaded preprocessed data:")
    print(f"   🎯 Training samples: {len(X_train)}")
    print(f"   🎯 Testing samples: {len(X_test)}")
    print(f"   📐 Feature shape: {X_train.shape[1:]}")
    print(f"   🏷️  Unique labels: {np.unique(y_train)}")
    
    return X_train, y_train, X_test, y_test

def create_model_with_dropout(dropout_rates):
    """Creates model with configurable dropout rates for different phases."""
    model = Sequential()
    
    # First Conv Block
    model.add(Conv1D(64, 3, padding='same', input_shape=(136,1), kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(dropout_rates[0]))
    
    # Second Conv Block
    model.add(Conv1D(128, 3, padding='same', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(dropout_rates[1]))
    
    # Third Conv Block
    model.add(Conv1D(256, 3, padding='same', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(dropout_rates[2]))
    
    # Fourth Conv Block
    model.add(Conv1D(512, 3, padding='same', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(dropout_rates[3]))
    
    # Dense Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout_rates[4]))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout_rates[5]))
    model.add(Dense(2, activation='softmax'))
    
    return model

def train_phase(model, X_train, y_train, X_test, y_test, phase_config, phase_num):
    """Train model for one phase with comprehensive monitoring."""
    
    phase_name = phase_config['name']
    print_phase_banner(phase_num, phase_name)
    
    # Display phase configuration
    config_info = f"""
🔧 PHASE {phase_num} CONFIGURATION:
   📈 Learning Rate: {phase_config['learning_rate']}
   📦 Batch Size: {phase_config['batch_size']}
   🔄 Epochs: {phase_config['epochs']}
   🎯 Dropout Rates: {phase_config['dropout_rates']}
   🎯 Focus: {phase_config['focus']}
"""
    print(config_info)
    
    # Compile model with new learning rate
    optimizer = Adam(learning_rate=phase_config['learning_rate'])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    # Setup callbacks
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"phase_{phase_num}_best_model.h5")
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1,
            save_format='h5'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print(f"🚀 Starting training for Phase {phase_num}...")
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=phase_config['epochs'],
        batch_size=phase_config['batch_size'],
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate phase results
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
    
    results_info = f"""
📊 PHASE {phase_num} RESULTS:
   ⏱️  Training Time: {training_time:.1f} seconds
   🎯 Final Training Accuracy: {train_acc:.4f}
   🎯 Final Validation Accuracy: {val_acc:.4f}
   📉 Final Training Loss: {train_loss:.4f}
   📉 Final Validation Loss: {val_loss:.4f}
   💾 Best model saved to: {checkpoint_path}
"""
    print(results_info)
    
    return history, model, {
        'phase': phase_num,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'training_time': training_time
    }

def plot_comprehensive_history(all_histories, all_results):
    """Create comprehensive training visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('4-Phase Drowsiness Detection Training Analysis', fontsize=16, fontweight='bold')
    
    # Combine all histories
    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    epoch_markers = [0]  # Phase boundaries
    
    for history in all_histories:
        train_acc.extend(history.history['accuracy'])
        val_acc.extend(history.history['val_accuracy'])
        train_loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
        epoch_markers.append(len(train_acc))
    
    # Plot 1: Accuracy over time
    axes[0, 0].plot(train_acc, label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(val_acc, label='Validation Accuracy', linewidth=2)
    for i, marker in enumerate(epoch_markers[1:-1]):
        axes[0, 0].axvline(x=marker, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].text(marker, 0.5, f'Phase {i+2}', rotation=90, alpha=0.7)
    axes[0, 0].set_title('Model Accuracy Across All Phases')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss over time
    axes[0, 1].plot(train_loss, label='Training Loss', linewidth=2)
    axes[0, 1].plot(val_loss, label='Validation Loss', linewidth=2)
    for i, marker in enumerate(epoch_markers[1:-1]):
        axes[0, 1].axvline(x=marker, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Model Loss Across All Phases')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Phase comparison - Validation Accuracy
    phases = [f"Phase {r['phase']}" for r in all_results]
    val_accs = [r['val_acc'] for r in all_results]
    bars = axes[0, 2].bar(phases, val_accs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 2].set_title('Validation Accuracy by Phase')
    axes[0, 2].set_ylabel('Accuracy')
    for bar, acc in zip(bars, val_accs):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                       f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Training Time by Phase
    training_times = [r['training_time'] for r in all_results]
    bars = axes[1, 0].bar(phases, training_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[1, 0].set_title('Training Time by Phase')
    axes[1, 0].set_ylabel('Time (seconds)')
    for bar, time_val in zip(bars, training_times):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Learning Rate Schedule
    learning_rates = [0.001, 0.0003, 0.0001, 1e-5]
    axes[1, 1].plot(range(1, 5), learning_rates, marker='o', linewidth=3, markersize=8)
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Phase')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(range(1, 5))
    
    # Plot 6: Batch Size Schedule
    batch_sizes = [64, 128, 256, 512]
    axes[1, 2].plot(range(1, 5), batch_sizes, marker='s', linewidth=3, markersize=8, color='orange')
    axes[1, 2].set_title('Batch Size Schedule')
    axes[1, 2].set_xlabel('Phase')
    axes[1, 2].set_ylabel('Batch Size')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xticks(range(1, 5))
    
    plt.tight_layout()
    plt.show()

def print_final_summary(all_results):
    """Print comprehensive final summary."""
    print("\n" + "="*80)
    print("🎉 FINAL TRAINING SUMMARY".center(80))
    print("="*80)
    
    best_phase = max(all_results, key=lambda x: x['val_acc'])
    total_time = sum(r['training_time'] for r in all_results)
    
    summary = f"""
📈 PERFORMANCE METRICS:
   🏆 Best Phase: Phase {best_phase['phase']} (Validation Accuracy: {best_phase['val_acc']:.4f})
   📊 Final Validation Accuracy: {all_results[-1]['val_acc']:.4f}
   📉 Final Training Loss: {all_results[-1]['train_loss']:.4f}
   ⏱️  Total Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)

📋 PHASE-BY-PHASE BREAKDOWN:"""
    print(summary)
    
    for result in all_results:
        phase_summary = f"""   Phase {result['phase']}: Val Acc = {result['val_acc']:.4f}, Time = {result['training_time']:.1f}s"""
        print(phase_summary)
    
    print(f"\n💾 Final model saved to: {MODEL_SAVE_NAME}")
    print("="*80)

def main():
    """Main training pipeline."""
    print("🚀 MULTI-PHASE DROWSINESS DETECTION TRAINING")
    print("="*80)
    print(f"⏰ Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load preprocessed data (much faster than raw image processing)
    print("\n📊 DATA LOADING PHASE")
    print("-" * 40)
    X_train, y_train, X_test, y_test = load_preprocessed_data()
    
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("❌ No valid data found. Check preprocessed data files.")
    
    # Reshape and encode
    X_train = X_train.reshape(-1, 136, 1)
    X_test = X_test.reshape(-1, 136, 1)
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)
    
    data_summary = f"""
📋 DATASET SUMMARY:
   🎯 Training samples: {len(X_train)}
   🎯 Testing samples: {len(X_test)}
   📐 Feature shape: {X_train.shape[1:]}
   🏷️  Classes: {len(np.unique(y_train))}
   📊 Label distribution: Train - {dict(zip(*np.unique(y_train, return_counts=True)))}
                          Test - {dict(zip(*np.unique(y_test, return_counts=True)))}
"""
    print(data_summary)
    
    # Define 4-phase training configuration
    training_phases = [
        {
            'name': 'Initial Learning',
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 18,
            'dropout_rates': (0.4, 0.4, 0.4, 0.4, 0.4, 0.4),
            'focus': 'Learn basic feature representations with higher learning rate'
        },
        {
            'name': 'Feature Refinement',
            'learning_rate': 0.0003,
            'batch_size': 128,
            'epochs': 12,
            'dropout_rates': (0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
            'focus': 'Refine features with moderate learning rate and larger batches'
        },
        {
            'name': 'Fine Tuning',
            'learning_rate': 0.0001,
            'batch_size': 256,
            'epochs': 8,
            'dropout_rates': (0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
            'focus': 'Fine-tune with low learning rate for precision'
        },
        {
            'name': 'Final Optimization',
            'learning_rate': 1e-5,
            'batch_size': 512,
            'epochs': 4,
            'dropout_rates': (0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
            'focus': 'Final optimization with minimal changes'
        }
    ]
    
    # Initialize model for first phase
    model = create_model_with_dropout(training_phases[0]['dropout_rates'])
    all_histories = []
    all_results = []
    
    # Execute training phases
    for phase_num, phase_config in enumerate(training_phases, 1):
        # Create new model with updated dropout rates for phases 2-4
        if phase_num > 1:
            # Save current weights
            temp_weights_path = os.path.join(CHECKPOINT_DIR, "temp_transfer_weights.h5")
            model.save_weights(temp_weights_path)
            
            # Create new model with updated dropout rates
            model = create_model_with_dropout(phase_config['dropout_rates'])
            
            # Load previous weights (architecture should be compatible)
            try:
                model.load_weights(temp_weights_path)
                print(f"✅ Successfully transferred weights from Phase {phase_num-1}")
            except Exception as e:
                print(f"⚠️  Warning: Could not transfer weights ({e}). Starting fresh for Phase {phase_num}")
            
            # Clean up temporary file
            if os.path.exists(temp_weights_path):
                os.remove(temp_weights_path)
        
        # Train current phase
        history, model, results = train_phase(
            model, X_train, y_train_cat, X_test, y_test_cat, 
            phase_config, phase_num
        )
        
        all_histories.append(history)
        all_results.append(results)
    
    # Save final model
    print(f"\n💾 Saving final model to: {MODEL_SAVE_NAME}")
    model.save(MODEL_SAVE_NAME)
    
    # Generate comprehensive visualizations and summary
    plot_comprehensive_history(all_histories, all_results)
    print_final_summary(all_results)
    
    print(f"\n🎉 Training completed successfully at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()