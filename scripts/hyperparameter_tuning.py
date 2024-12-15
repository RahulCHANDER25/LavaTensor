#!/usr/bin/env python3
import optuna
import subprocess
import json
import os
import re
from typing import Dict, Any
import configparser
import logging
import datetime
import sys

N_TRIALS = 1000
EPOCHS = 100

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_network_config(trial: optuna.Trial, base_config_path: str, output_path: str) -> None:
    config = configparser.ConfigParser()
    config.read(base_config_path)
    params_for_logging = {}

    # Hyperparameters to tune
    params_for_logging['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)

    # Batch size as multiple of 16 (from 32 to 256)
    batch_size_multiplier = trial.suggest_int('batch_size_multiplier', 2, 16)  # 2*16=32 to 16*16=256
    params_for_logging['batch_size'] = batch_size_multiplier * 16

    params_for_logging['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
    params_for_logging['decay_rate'] = trial.suggest_float('decay_rate', 0.8, 0.99)
    params_for_logging['decay_steps'] = trial.suggest_int('decay_steps', 1, 10)
    params_for_logging['min_lr'] = trial.suggest_float('min_lr', 1e-5, 1e-3, log=True)

    # Training parameters
    params_for_logging['epochs'] = EPOCHS
    # Samples per epoch as multiple of batch size
    samples_multiplier = trial.suggest_int('samples_per_epoch_multiplier', 8, 32)  # 8*batch_size to 64*batch_size
    params_for_logging['samples_per_epoch'] = samples_multiplier * params_for_logging['batch_size']

    config['hyperparameters']['learning_rate'] = str(params_for_logging['learning_rate'])
    config['hyperparameters']['batch_size'] = str(params_for_logging['batch_size'])
    config['hyperparameters']['dropout'] = str(params_for_logging['dropout'])
    config['hyperparameters']['epochs'] = str(params_for_logging['epochs'])
    config['hyperparameters']['samples_per_epoch'] = str(params_for_logging['samples_per_epoch'])

    config['lr_scheduler']['type'] = "exponential"
    config['lr_scheduler']['decay_rate'] = str(params_for_logging['decay_rate'])
    config['lr_scheduler']['decay_steps'] = str(params_for_logging['decay_steps'])
    config['lr_scheduler']['min_lr'] = str(params_for_logging['min_lr'])

    n_layers = trial.suggest_int('n_hidden_layers', 2, 5)
    params_for_logging['n_hidden_layers'] = n_layers
    hidden_sizes = []

    # Calculate valid power of 2 sizes (all greater than output size 6)
    min_power = 3  # 2^3 = 8 (first power of 2 > 6)
    max_power = 11  # 2^11 = 2048
    valid_sizes = [2**i for i in range(min_power, max_power + 1)]


    size_idx = trial.suggest_int('hidden_size_0', 0, len(valid_sizes)-1)
    first_layer_size = valid_sizes[size_idx]
    hidden_sizes.append(str(first_layer_size))
    params_for_logging['hidden_size_0'] = first_layer_size

    prev_size_idx = valid_sizes.index(first_layer_size)
    for i in range(1, n_layers):
        # Get valid indices for this layer (must be <= previous layer's size)
        valid_indices = list(range(0, prev_size_idx + 1))
        if not valid_indices:
            size_idx = 0
        else:
            size_idx = trial.suggest_int(f'hidden_size_{i}', 0, len(valid_indices)-1)
            size_idx = valid_indices[size_idx]

        layer_size = valid_sizes[size_idx]
        hidden_sizes.append(str(layer_size))
        params_for_logging[f'hidden_size_{i}'] = layer_size
        prev_size_idx = size_idx

    config['architecture']['hidden_layers'] = str(n_layers)
    config['architecture']['hidden_sizes'] = ','.join(hidden_sizes)

    with open(output_path, 'w') as f:
        for section in config.sections():
            f.write(f'[{section}]\n')
            for key, value in config[section].items():
                f.write(f'{key}={value}\n')
            f.write('\n')

    logger.info(f"Created configuration file: {output_path}")
    logger.info("⚙️ Configuration Parameters:")
    for key, value in params_for_logging.items():
        if key.startswith('hidden_size'):
            logger.info(f"  • {key}: {value} neurons")
        else:
            logger.info(f"  • {key}: {value}")
    return params_for_logging

def extract_epoch_metrics(line: str) -> Dict[str, float]:
    epoch_pattern = r'Epoch (\d+)/\d+ \(\d+ samples\) - Loss: ([\d.]+) - Accuracy: ([\d.]+)% - LR: ([\d.e-]+)'
    match = re.match(epoch_pattern, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'loss': float(match.group(2)),
            'accuracy': float(match.group(3)),
            'lr': float(match.group(4))
        }
    return None

def print_epoch_row(epoch: int, loss: float, accuracy: float, lr: float,
                   best_acc: float = None, best_loss: float = None) -> None:
    acc_indicator = "🔥" if best_acc is not None and accuracy >= best_acc else " "
    loss_indicator = "✨" if best_loss is not None and loss <= best_loss else " "
    row = f"{epoch:^6} │ {loss:^10.4f} │ {accuracy:^9.2f}% │ {lr:^10.2e} {acc_indicator}{loss_indicator}"
    logger.info(row)

def train_network(network_path: str, config_path: str, trained_path: str) -> Dict[str, float]:
    metrics = {
        'accuracy_history': [],
        'loss_history': [],
        'lr_history': []
    }

    process = subprocess.Popen(
        ['./my_torch_analyzer', '--train', '--save', trained_path, network_path, 'examples/training_positions.txt'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    logger.info("\n📊 Training Progress")
    logger.info("─" * 60)
    logger.info(f"{'Epoch':^6} │ {'Loss':^10} │ {'Accuracy':^10} │ {'LR':^10}")
    logger.info("─" * 60)

    best_acc = float('-inf')
    best_loss = float('inf')


    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break

        epoch_metrics = extract_epoch_metrics(line.strip())
        if epoch_metrics:

            metrics['accuracy_history'].append(epoch_metrics['accuracy'])
            metrics['loss_history'].append(epoch_metrics['loss'])
            metrics['lr_history'].append(epoch_metrics['lr'])

            best_acc = max(best_acc, epoch_metrics['accuracy'])
            best_loss = min(best_loss, epoch_metrics['loss'])

            print_epoch_row(
                epoch_metrics['epoch'],
                epoch_metrics['loss'],
                epoch_metrics['accuracy'],
                epoch_metrics['lr'],
                best_acc,
                best_loss
            )

    return_code = process.poll()
    error_output = process.stderr.read()

    if return_code != 0:
        logger.error(f"❌ Training failed with return code: {return_code}")
        logger.error(f"Error output: {error_output}")
        return {}

    if metrics['accuracy_history']:
        metrics['final_accuracy'] = metrics['accuracy_history'][-1]
        metrics['final_loss'] = metrics['loss_history'][-1]
        metrics['final_lr'] = metrics['lr_history'][-1]

        logger.info("─" * 60)

        logger.info("\n📈 Final Results:")
        logger.info(f"  📌 Accuracy: {metrics['final_accuracy']:.2f}%")
        logger.info(f"  📉 Loss: {metrics['final_loss']:.4f}")
        logger.info(f"  📊 Learning Rate: {metrics['final_lr']:.2e}")

        first_acc = metrics['accuracy_history'][0]
        acc_improvement = metrics['final_accuracy'] - first_acc
        logger.info(f"\n📊 Training Summary:")
        logger.info(f"  📈 Accuracy Improvement: {acc_improvement:+.2f}%")
        logger.info(f"  🎯 Best Accuracy: {best_acc:.2f}%")
        logger.info(f"  📉 Best Loss: {best_loss:.4f}")

    return metrics

def objective(trial: optuna.Trial) -> float:
    trial_config_path = f'examples/trial_{trial.number}.conf'
    network_path = f'trial_{trial.number}_1.nn'
    trained_path = f'models/trial_{trial.number}.nn'

    try:
        logger.info(f"\n{'═'*80}")
        logger.info(f"🔄 Starting Trial {trial.number}")
        logger.info(f"{'═'*80}")

        create_network_config(trial, 'examples/basic_network.conf', trial_config_path)

        logger.info(f"\n🔨 Generating network for trial {trial.number}")
        gen_result = subprocess.run(
            ['./my_torch_generator', trial_config_path, '1'],
            capture_output=True,
            text=True
        )

        if gen_result.returncode != 0:
            logger.error(f"❌ Network generation failed: {gen_result.stderr}")
            logger.error(f"Generator output: {gen_result.stdout}")
            return float('inf')

        if not os.path.exists(network_path):
            logger.error(f"❌ Network file was not generated: {network_path}")
            return float('inf')

        logger.info(f"✅ Network generated successfully: {network_path}")

        logger.info(f"\n🚀 Starting training for trial {trial.number}")
        metrics = train_network(network_path, trial_config_path, trained_path)

        os.remove(trial_config_path)
        os.remove(network_path)

        if not metrics or 'final_accuracy' not in metrics:
            logger.error("❌ No accuracy metric found in output")
            return float('inf')

        accuracy = metrics['final_accuracy']
        logger.info(f"\n✨ Trial {trial.number} Results:")
        logger.info(f"  📊 Final accuracy: {accuracy}%")
        if 'final_loss' in metrics:
            logger.info(f"  📉 Final loss: {metrics['final_loss']}")

        try:
            best_trial = trial.study.best_trial
            best_value = -best_trial.value
            if accuracy > best_value:
                logger.info(f"  🏆 New best trial! Previous best: {best_value:.2f}%")
            else:
                logger.info(f"  ℹ️ Best accuracy so far: {best_value:.2f}%")
        except ValueError:
            logger.info("  🎯 First trial completed")
        logger.info(f"{'═'*80}\n")
        return -accuracy

    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Trial {trial.number} timed out")
        return float('inf')
    except Exception as e:
        logger.error(f"❌ Error in trial {trial.number}: {str(e)}")
        logger.exception("Detailed error information:")
        return float('inf')
    finally:
        for file in [trial_config_path, network_path]:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    logger.info(f"🧹 Cleaned up file: {file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up file {file}: {e}")

def main():
    os.makedirs('models', exist_ok=True)

    study_name = "chess_network_optimization"
    storage_name = "sqlite:///chess_optimization.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )

    logger.info("\n🚀 Starting hyperparameter optimization")
    logger.info(f"📂 Training data: examples/training_positions.txt")
    logger.info(f"📄 Base config: examples/basic_network.conf")
    logger.info(f"💾 Results will be saved in: models/")
    logger.info(f"{'─'*80}\n")

    n_trials = N_TRIALS
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")

    if study.best_trial:
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {-trial.value:.2f}% accuracy")
        print("\nBest hyperparameters:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")
        best_config_path = 'examples/best_network.conf'
        create_network_config(trial, 'examples/basic_network.conf', best_config_path)
        print(f"\nBest configuration saved to {best_config_path}")

        try:
            param_importance = optuna.importance.get_param_importances(study)
        except ImportError:
            logger.warning("scikit-learn not installed. Skipping parameter importance analysis.")
            param_importance = None

        report = {
            'best_accuracy': -trial.value,
            'best_params': trial.params,
            'n_trials': n_trials,
            'optimization_history': [
                {'trial': t.number, 'accuracy': -t.value, 'params': t.params}
                for t in study.trials if t.value is not None
            ],
            'importance': param_importance,
            'system_info': {
                'date': str(datetime.datetime.now()),
                'python_version': sys.version,
                'optuna_version': optuna.__version__
            }
        }

        report_path = 'hyperparameter_optimization_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to {report_path}")
    else:
        logger.error("No successful trials completed")

if __name__ == "__main__":
    main()