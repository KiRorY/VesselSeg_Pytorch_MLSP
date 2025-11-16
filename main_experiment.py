# -*- coding: utf-8 -*-
"""
Main Experiment Script: Train and Compare All Models

This script:
1. Loads and preprocesses DRIVE dataset
2. Trains all models (U-Net, U-Net+EAM, TransUnet, TransUnet+EAM)
3. Evaluates each model comprehensively
4. Performs statistical comparison
5. Generates visualizations and reports
"""

import os
import sys
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

# Import custom modules
from trainer import (load_raw_drive_data, generate_patches, VesselPatchDataset,
                     train_model, evaluate_model)
from models import get_model
from metrics import MetricsCalculator
from visualization import (plot_training_curves, plot_roc_curves, plot_metrics_comparison,
                           create_metrics_table, plot_radar_chart, plot_segmentation_results,
                           plot_error_maps)
from comparison import (pairwise_comparison, create_summary_table, rank_models,
                       save_results, generate_text_report)


# ============================================================================
# Configuration
# ============================================================================

def get_config():
    """Get experiment configuration"""
    config = {
        # Data paths
        "raw_drive_path": "./datasets/DRIVE",

        # Patch generation
        "patch_height": 64,
        "patch_width": 64,
        "n_patches": 150000,
        "val_ratio": 0.1,

        # Model settings
        "in_channels": 1,
        "classes": 2,

        # Training hyperparameters
        "epochs": 25,
        "batch_size": 64,
        "lr": 0.0005,
        "early_stop": 6,

        # Loss function
        "loss": "focal_dice",  # Options: 'ce', 'dice', 'bce_dice', 'focal_dice'

        # Deep supervision (for EAM models)
        "deep_supervision": True,

        # Output directories
        "checkpoint_dir": "checkpoints",
        "results_dir": "results",
        "figures_dir": "figures",

        # Random seed
        "seed": 2025,

        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # Models to train
        "models_to_train": ['unet', 'unet_eam', 'transunet', 'transunet_eam']
    }

    return config


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data(config):
    """
    Load and prepare dataset

    Returns:
        train_loader, val_loader
    """
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION")
    print("="*80)

    # Load raw data
    print("\n1.1 Loading raw DRIVE data...")
    images, masks, fovs = load_raw_drive_data(config['raw_drive_path'])

    if images is None:
        print("ERROR: Failed to load data. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(images)} images successfully.")

    # Generate patches
    print("\n1.2 Generating patches...")
    patch_data = generate_patches(
        images, masks, fovs,
        (config['patch_height'], config['patch_width']),
        config['n_patches'],
        seed=config['seed']
    )

    print(f"Generated {len(patch_data)} patches.")

    # Split into train and validation
    print(f"\n1.3 Splitting data ({config['val_ratio']*100}% for validation)...")
    from sklearn.model_selection import train_test_split

    train_patches, val_patches = train_test_split(
        patch_data,
        test_size=config['val_ratio'],
        random_state=config['seed']
    )

    # Create datasets
    train_dataset = VesselPatchDataset(train_patches, augment=True)
    val_dataset = VesselPatchDataset(val_patches, augment=False)

    # Create dataloaders
    num_workers = max(1, os.cpu_count() // 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nTrain set: {len(train_dataset)} patches, {len(train_loader)} batches")
    print(f"Val set:   {len(val_dataset)} patches, {len(val_loader)} batches")

    return train_loader, val_loader, val_dataset


# ============================================================================
# Train All Models
# ============================================================================

def train_all_models(config, train_loader, val_loader):
    """
    Train all models specified in config

    Returns:
        trained_models: dict of {model_name: model}
        histories: dict of {model_name: training_history}
    """
    print("\n" + "="*80)
    print("STEP 2: TRAINING ALL MODELS")
    print("="*80)

    device = config['device']
    models_to_train = config['models_to_train']

    trained_models = {}
    histories = {}
    best_metrics_dict = {}

    for model_name in models_to_train:
        print(f"\n{'='*80}")
        print(f"Training: {model_name.upper()}")
        print(f"{'='*80}")

        # Adjust config for EAM models
        model_config = config.copy()
        if 'eam' in model_name.lower():
            model_config['deep_supervision'] = True
        else:
            model_config['deep_supervision'] = False

        # Train model
        start_time = time.time()
        model, history, best_metrics = train_model(
            model_name,
            model_config,
            train_loader,
            val_loader,
            device,
            save_dir=config['checkpoint_dir']
        )
        elapsed = time.time() - start_time

        print(f"\nTraining completed in {elapsed/60:.2f} minutes")

        trained_models[model_name] = model
        histories[model_name] = history
        best_metrics_dict[model_name] = best_metrics

    return trained_models, histories, best_metrics_dict


# ============================================================================
# Evaluate All Models
# ============================================================================

def evaluate_all_models(trained_models, val_loader, config):
    """
    Evaluate all trained models comprehensively

    Returns:
        all_metrics: dict of {model_name: average_metrics}
        all_results: dict of {model_name: detailed_results}
        all_calculators: dict of {model_name: MetricsCalculator}
    """
    print("\n" + "="*80)
    print("STEP 3: COMPREHENSIVE EVALUATION")
    print("="*80)

    device = config['device']

    all_metrics = {}
    all_results = {}
    all_calculators = {}

    for model_name, model in trained_models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'='*60}")

        metrics, calculator = evaluate_model(model, val_loader, device)

        all_metrics[model_name] = metrics
        all_calculators[model_name] = calculator

        # Also store per-sample results for statistical tests
        all_results[model_name] = {
            'dice': calculator.metrics['dice'],
            'iou': calculator.metrics['iou'],
            'sensitivity': calculator.metrics['sensitivity'],
            'specificity': calculator.metrics['specificity'],
            'precision': calculator.metrics['precision'],
            'boundary_f1': calculator.metrics['boundary_f1'],
            'assd': calculator.metrics['assd']
        }

        # Print summary
        print(f"\nResults for {model_name}:")
        print(f"  Dice:        {metrics['dice']:.4f} ± {metrics['dice_std']:.4f}")
        print(f"  IoU:         {metrics['iou']:.4f} ± {metrics['iou_std']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f} ± {metrics['sensitivity_std']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f} ± {metrics['specificity_std']:.4f}")
        print(f"  Boundary F1: {metrics['boundary_f1']:.4f} ± {metrics['boundary_f1_std']:.4f}")
        print(f"  ASSD:        {metrics['assd']:.4f} ± {metrics['assd_std']:.4f}")
        print(f"  AUC:         {metrics['auc']:.4f}")

    return all_metrics, all_results, all_calculators


# ============================================================================
# Generate Visualizations
# ============================================================================

def generate_visualizations(histories, all_metrics, all_calculators, trained_models,
                           val_dataset, config):
    """Generate all visualization plots"""
    print("\n" + "="*80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*80)

    figures_dir = config['figures_dir']
    os.makedirs(figures_dir, exist_ok=True)

    device = config['device']

    # 1. Training curves
    print("\n4.1 Plotting training curves...")
    plot_training_curves(histories, os.path.join(figures_dir, 'training_curves.png'))

    # 2. Metrics comparison
    print("4.2 Plotting metrics comparison...")
    plot_metrics_comparison(all_metrics, os.path.join(figures_dir, 'metrics_comparison.png'))

    # 3. ROC curves
    print("4.3 Plotting ROC curves...")
    roc_data = {}
    for model_name, calculator in all_calculators.items():
        fpr, tpr, auc_score = calculator.get_roc_curve()
        if fpr is not None:
            roc_data[model_name] = (fpr, tpr, auc_score)

    if roc_data:
        plot_roc_curves(roc_data, os.path.join(figures_dir, 'roc_curves.png'))

    # 4. Radar chart
    print("4.4 Plotting radar chart...")
    plot_radar_chart(all_metrics, os.path.join(figures_dir, 'radar_chart.png'))

    # 5. Metrics table visualization
    print("4.5 Creating metrics table...")
    create_metrics_table(all_metrics, os.path.join(figures_dir, 'metrics_table.png'))

    # 6. Segmentation results visualization
    print("4.6 Generating segmentation results...")
    num_samples = min(5, len(val_dataset))

    # Get random samples
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)

    images = []
    ground_truths = []
    predictions = {model_name: [] for model_name in trained_models.keys()}

    for idx in indices:
        img_tensor, mask = val_dataset[idx]
        img_np = img_tensor.squeeze().numpy()
        mask_np = mask.numpy()

        images.append(img_np)
        ground_truths.append(mask_np)

        # Get predictions from each model
        for model_name, model in trained_models.items():
            model.eval()
            with torch.no_grad():
                img_input = img_tensor.unsqueeze(0).to(device)
                output = model(img_input)

                if isinstance(output, tuple):
                    output = output[0]

                pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                predictions[model_name].append(pred)

    plot_segmentation_results(
        images, ground_truths, predictions, list(trained_models.keys()),
        num_samples=num_samples,
        save_path=os.path.join(figures_dir, 'segmentation_results.png')
    )

    # 7. Error maps
    print("4.7 Generating error maps...")
    # Use first sample for error map
    img = images[0]
    gt = ground_truths[0]
    preds = {model_name: predictions[model_name][0] for model_name in trained_models.keys()}

    plot_error_maps(
        img, gt, preds, list(trained_models.keys()),
        save_path=os.path.join(figures_dir, 'error_maps.png')
    )

    print(f"\nAll visualizations saved to {figures_dir}/")


# ============================================================================
# Statistical Analysis
# ============================================================================

def perform_statistical_analysis(all_metrics, all_results, config):
    """Perform statistical comparison and generate reports"""
    print("\n" + "="*80)
    print("STEP 5: STATISTICAL ANALYSIS")
    print("="*80)

    results_dir = config['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # 1. Pairwise comparison
    print("\n5.1 Performing pairwise statistical comparison...")
    comparison_results = pairwise_comparison(all_results)

    # 2. Create summary tables
    print("\n5.2 Creating summary tables...")
    summary_df = create_summary_table(
        all_metrics,
        save_path=os.path.join(results_dir, 'summary_table.csv')
    )

    # 3. Rank models
    print("\n5.3 Ranking models...")
    ranks = rank_models(all_metrics)
    ranks.to_csv(os.path.join(results_dir, 'model_rankings.csv'))

    # 4. Save all results
    print("\n5.4 Saving results...")
    save_results(all_metrics, all_results, comparison_results, results_dir)

    # 5. Generate comprehensive report
    print("\n5.5 Generating comprehensive report...")
    generate_text_report(
        all_metrics, all_results, comparison_results,
        save_path=os.path.join(results_dir, 'comprehensive_report.txt')
    )

    return comparison_results


# ============================================================================
# Main Experiment Function
# ============================================================================

def run_experiment(config):
    """Run complete experiment pipeline"""
    print("\n" + "="*80)
    print("MEDICAL IMAGE SEGMENTATION - MODEL COMPARISON EXPERIMENT")
    print("="*80)
    print(f"\nDevice: {config['device']}")
    print(f"Models to train: {', '.join(config['models_to_train'])}")
    print(f"Loss function: {config['loss']}")
    print(f"Deep supervision: {config['deep_supervision']}")

    # Early stopping info
    if config['early_stop'] == float('inf'):
        print(f"Early stopping: DISABLED (will train for all {config['epochs']} epochs)")
    else:
        print(f"Early stopping: ENABLED (patience={config['early_stop']} epochs)")

    # Set seed for reproducibility
    set_seed(config['seed'])

    # Create output directories
    for dir_name in ['checkpoint_dir', 'results_dir', 'figures_dir']:
        os.makedirs(config[dir_name], exist_ok=True)

    # Step 1: Prepare data
    train_loader, val_loader, val_dataset = prepare_data(config)

    # Step 2: Train all models
    trained_models, histories, best_metrics_dict = train_all_models(
        config, train_loader, val_loader
    )

    # Step 3: Evaluate all models
    all_metrics, all_results, all_calculators = evaluate_all_models(
        trained_models, val_loader, config
    )

    # Step 4: Generate visualizations
    generate_visualizations(
        histories, all_metrics, all_calculators,
        trained_models, val_dataset, config
    )

    # Step 5: Statistical analysis
    comparison_results = perform_statistical_analysis(
        all_metrics, all_results, config
    )

    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nSummary:")
    print(f"  Checkpoints saved to: {config['checkpoint_dir']}/")
    print(f"  Results saved to:     {config['results_dir']}/")
    print(f"  Figures saved to:     {config['figures_dir']}/")
    print("\nBest performing model:")

    best_model = max(all_metrics.items(), key=lambda x: x[1]['dice'])
    print(f"  {best_model[0].upper()}")
    print(f"  Dice Score: {best_model[1]['dice']:.4f}")
    print(f"  IoU:        {best_model[1]['iou']:.4f}")
    print(f"  AUC:        {best_model[1]['auc']:.4f}")

    print("\n" + "="*80)

    return trained_models, all_metrics, all_results


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Train and compare medical image segmentation models'
    )

    parser.add_argument('--data-path', type=str,
                       default='./datasets/DRIVE',
                       help='Path to DRIVE dataset')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--loss', type=str, default='focal_dice',
                       choices=['ce', 'dice', 'bce_dice', 'focal_dice'],
                       help='Loss function')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['unet', 'unet_eam', 'transunet', 'transunet_eam'],
                       help='Models to train')
    parser.add_argument('--seed', type=int, default=2025,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for results')
    parser.add_argument('--early-stop', type=int, default=3,
                       help='Early stopping patience (epochs without improvement). Default: 6')
    parser.add_argument('--disable-early-stop', action='store_true',
                       help='Disable early stopping (train for all epochs)')

    args = parser.parse_args()

    # Get default config and update with arguments
    config = get_config()
    config['raw_drive_path'] = args.data_path
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['loss'] = args.loss
    config['models_to_train'] = args.models
    config['seed'] = args.seed

    # Early stopping configuration
    if args.disable_early_stop:
        config['early_stop'] = float('inf')  # Never trigger early stopping
    else:
        config['early_stop'] = args.early_stop

    # Update output directories
    config['checkpoint_dir'] = os.path.join(args.output_dir, 'checkpoints')
    config['results_dir'] = os.path.join(args.output_dir, 'results')
    config['figures_dir'] = os.path.join(args.output_dir, 'figures')

    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()
