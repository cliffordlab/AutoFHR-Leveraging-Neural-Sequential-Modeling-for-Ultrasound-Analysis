import os
import sys
import argparse
import os
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model import AutoFHRModel
from src.models.losses import introduced_loss
from src.models.trainer import ModelTrainer
from src.data.data_loader import AutoFHRDataLoader, AutoFHRDataGenerator
from src.utils.visualization import (
    bland_altman_plot, scatter_plot_with_regression, 
    attention_plot, time_series_robustness_plot,
    plot_predictions_with_ground_truth
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train the AutoFHR model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset file (.npz)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    
    # Model parameters
    parser.add_argument('--filters', type=int, default=40,
                        help='Number of filters')
    parser.add_argument('--kernel_size', type=int, default=20,
                        help='Kernel size')
    parser.add_argument('--dilation_layers', type=int, default=8,
                        help='Number of dilation layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=100,
                        help='Patience for early stopping')
    parser.add_argument('--use_generator', action='store_true',
                        help='Use data generator for training')
    
    # Visualization parameters
    parser.add_argument('--num_vis_samples', type=int, default=4,
                        help='Number of samples to visualize')
    parser.add_argument('--vis_attention', action='store_true',
                        help='Visualize attention weights')
    
    # Other parameters
    parser.add_argument('--random_seed', type=int, default=2025,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device to use (use -1 for CPU)')
    
    return parser.parse_args()

def setup_gpu(gpu_id):
    """Set up GPU device"""
    if gpu_id >= 0:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                # Allow memory growth for the GPU
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
                print(f"Using GPU {gpu_id}: {gpus[gpu_id]}")
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")
        else:
            print("No GPUs found. Using CPU instead.")
    else:
        tf.config.set_visible_devices([], 'GPU')
        print("Using CPU as specified.")

def main():
    """Main function to train the AutoFHR model"""
    args = parse_args()
    
    # Set up GPU
    setup_gpu(args.gpu)
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    data_loader = AutoFHRDataLoader(args.data_path)
    train_data, train_labels, val_data, val_labels, test_data, test_labels = data_loader.split_data()
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    model_builder = AutoFHRModel(random_seed=args.random_seed)
    dilation_rates = [2**i for i in range(args.dilation_layers)]
    model = model_builder.build(
        input_shape=train_data.shape[1:],
        filters=args.filters,
        kernel_size=args.kernel_size,
        dilation_rates=dilation_rates,
        num_heads=args.num_heads
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        loss=introduced_loss,
        save_weights_only=False, 
        save_best_only=True,
        shuffle=True,
        metrics=['val_loss'],
        checkpoint_dir=os.path.join(output_dir, 'checkpoints')
    )
    
    # Set up callbacks
    callbacks = trainer.setup_callbacks(
        patience=args.patience,
        log_dir=os.path.join(output_dir, 'logs'),
        model_name="AutoFHR"
    )
    
    # Train the model
    if args.use_generator:
        train_generator = AutoFHRDataGenerator(train_data, train_labels, batch_size=args.batch_size)
        val_generator = AutoFHRDataGenerator(val_data, val_labels, batch_size=args.batch_size)

        history = trainer.train_with_generator(
            train_generator=train_generator,
            val_generator=val_generator,
            epochs=args.epochs,
            callbacks=callbacks
        )
    else:
        history = trainer.train(
            train_data=train_data,
            train_labels=train_labels,
            val_data=val_data,
            val_labels=val_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks
        )
    
    # Evaluate the model 
    test_loss, test_accuracy = trainer.evaluate(test_data, test_labels)
    print(f"Test loss: {test_loss:.4f}")
    
    # Make predictions 
    predictions = trainer.predict(test_data)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    print("Generating advanced visualizations...")

    test_predictions = trainer.predict(test_data)
    
    prediction_values = []
    ground_truth_values = []
    
    # Plot predictions with ground truth for selected samples
    num_vis_samples = min(args.num_vis_samples, len(test_data))
    sample_indices = np.random.choice(len(test_data), num_vis_samples, replace=False)
    
    sample_signals = test_data[sample_indices]
    sample_predictions = test_predictions[sample_indices]
    sample_ground_truths = test_labels[sample_indices]
    
    fig_preds, _ = plot_predictions_with_ground_truth(
        sample_signals,
        sample_predictions,
        sample_ground_truths,
        n_samples=num_vis_samples,
        save_path=os.path.join(vis_dir, 'predictions_ground_truth.png')
    )

    # Attention visualization (if model has attention mechanism and flag is set)
    if args.vis_attention and hasattr(model, 'get_attention_weights'):
        # Get attention weights for a sample
        sample_idx = sample_indices[0]
        sample_signal = test_data[sample_idx:sample_idx+1]
        
        # Get attention weights (assumes the model has a method to extract them)
        attention_weights = model.get_attention_weights(sample_signal)[0]
        
        fig_attention, _ = attention_plot(
            test_data[sample_idx],
            attention_weights,
            title="Attention Weights Visualization",
            save_path=os.path.join(vis_dir, 'attention_plot.png')
        )    
    
    # Save the best model
    best_model_path = os.path.join(output_dir, 'model', 'autofhr_model.h5')
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    trainer.save_model(best_model_path)
    
    print(f"Training completed. Results saved to {output_dir}")
    print(f"Visualizations saved to {vis_dir}")

if __name__ == '__main__':
    main() 