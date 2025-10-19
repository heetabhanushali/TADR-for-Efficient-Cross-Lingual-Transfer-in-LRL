"""
TADR Framework - Step 8: Zero-Shot Transfer Evaluation
Production-ready evaluation pipeline for cross-lingual zero-shot transfer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json
import os
import pickle
import pandas as pd
from tqdm import tqdm
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

# Import components from previous steps
from step5_integration import CompleteTADRModel, create_tadr_model
from step7_training_loop import MultilingualDataset, collate_fn, load_preprocessed_data
from transformers import AutoTokenizer

# ============================================================================
# EVALUATION METRICS
# ============================================================================

class ZeroShotEvaluator:
    """
    Comprehensive evaluator for zero-shot cross-lingual transfer.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = 'cuda',
        verbose: bool = True
    ):
        """
        Args:
            model: Trained TADR model
            tokenizer: Tokenizer for the model
            device: Device to use
            verbose: Whether to print progress
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.verbose = verbose
        
        # Results storage
        self.results = {
            'per_language': {},
            'overall': {},
            'routing_analysis': {},
            'typology_correlation': {},
            'error_analysis': {}
        }
    
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        dataset_name: str = "test",
        return_predictions: bool = True,
        analyze_routing: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate on a complete dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            dataset_name: Name of the dataset (for logging)
            return_predictions: Whether to return all predictions
            analyze_routing: Whether to analyze routing patterns
        
        Returns:
            Comprehensive evaluation results
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_languages = []
        all_routing_weights = []
        all_confidences = []
        
        # Per-language metrics
        lang_metrics = defaultdict(lambda: {
            'predictions': [],
            'labels': [],
            'correct': 0,
            'total': 0
        })
        
        with torch.no_grad():
            progress_bar = tqdm(
                dataloader,
                desc=f"Evaluating {dataset_name}",
                disable=not self.verbose
            )
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                language_ids = batch['language_ids']
                
                # Forward pass
                outputs = self.model(
                    lang_ids=language_ids,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_routing_info=analyze_routing
                )
                
                # Get predictions
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                confidences = torch.max(probs, dim=-1)[0]
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_languages.extend(language_ids)
                all_confidences.extend(confidences.cpu().numpy())
                
                # Store routing weights if analyzing
                if analyze_routing and 'routing_info' in outputs:
                    # Average routing weights across layers
                    layer_weights = [info['weights'] for info in outputs['routing_info']]
                    avg_weights = torch.stack(layer_weights).mean(dim=0)
                    all_routing_weights.append(avg_weights.cpu())
                
                # Update per-language metrics
                for i, lang_id in enumerate(language_ids):
                    pred = predictions[i].item()
                    label = labels[i].item()
                    
                    lang_metrics[lang_id]['predictions'].append(pred)
                    lang_metrics[lang_id]['labels'].append(label)
                    lang_metrics[lang_id]['total'] += 1
                    if pred == label:
                        lang_metrics[lang_id]['correct'] += 1
        
        # Compute overall metrics
        overall_metrics = self._compute_metrics(
            np.array(all_predictions),
            np.array(all_labels)
        )
        
        # Compute per-language metrics
        per_language_metrics = {}
        for lang_id, metrics in lang_metrics.items():
            per_language_metrics[lang_id] = self._compute_metrics(
                np.array(metrics['predictions']),
                np.array(metrics['labels'])
            )
        
        # Analyze routing patterns if requested
        routing_analysis = None
        if analyze_routing and all_routing_weights:
            routing_analysis = self._analyze_routing_patterns(
                torch.cat(all_routing_weights, dim=0),
                all_languages
            )
        
        results = {
            'overall': overall_metrics,
            'per_language': per_language_metrics,
            'routing_analysis': routing_analysis,
            'predictions': all_predictions if return_predictions else None,
            'labels': all_labels if return_predictions else None,
            'languages': all_languages if return_predictions else None,
            'confidences': all_confidences if return_predictions else None
        }
        
        return results
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics.
        
        Args:
            predictions: Predicted labels
            labels: True labels
        
        Returns:
            Dictionary of metrics
        """
        if len(predictions) == 0:
            return {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_micro': 0.0,
                'f1_weighted': 0.0,
                'n_samples': 0
            }
        
        accuracy = (predictions == labels).mean()
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'n_samples': len(predictions)
        }
    
    def _analyze_routing_patterns(
        self,
        routing_weights: torch.Tensor,
        languages: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze adapter routing patterns.
        
        Args:
            routing_weights: Routing weights (n_samples, n_adapters)
            languages: Language ID for each sample
        
        Returns:
            Routing analysis results
        """
        routing_weights = routing_weights.numpy()
        n_adapters = routing_weights.shape[1]
        
        # Per-language routing preferences
        lang_routing = defaultdict(list)
        for weights, lang in zip(routing_weights, languages):
            lang_routing[lang].append(weights)
        
        # Average routing per language
        avg_routing = {}
        for lang, weights_list in lang_routing.items():
            avg_routing[lang] = np.mean(weights_list, axis=0)
        
        # Compute routing entropy (diversity of adapter usage)
        epsilon = 1e-10
        routing_entropy = {}
        for lang, avg_weights in avg_routing.items():
            entropy = -np.sum(avg_weights * np.log(avg_weights + epsilon))
            routing_entropy[lang] = entropy
        
        # Most used adapters per language
        top_adapters = {}
        for lang, avg_weights in avg_routing.items():
            top_3 = np.argsort(avg_weights)[-3:][::-1]
            top_adapters[lang] = {
                'indices': top_3.tolist(),
                'weights': avg_weights[top_3].tolist()
            }
        
        return {
            'average_routing': avg_routing,
            'routing_entropy': routing_entropy,
            'top_adapters': top_adapters,
            'n_adapters': n_adapters
        }
    
    def evaluate_zero_shot_transfer(
        self,
        test_data: Dict,
        train_languages: List[str],
        batch_size: int = 32,
        max_length: int = 128
    ) -> Dict[str, Any]:
        """
        Main zero-shot evaluation function.
        
        Args:
            test_data: Test data dictionary
            train_languages: Languages seen during training
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
        
        Returns:
            Complete zero-shot evaluation results
        """
        print("\n" + "="*70)
        print("ZERO-SHOT CROSS-LINGUAL TRANSFER EVALUATION")
        print("="*70)
        
        # Create test dataloader
        test_dataset = MultilingualDataset(
            texts=test_data['texts'],
            labels=test_data['labels'],
            language_ids=test_data['lang_ids'],
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Identify seen vs unseen languages
        test_languages = list(set(test_data['lang_ids']))
        seen_languages = [lang for lang in test_languages if lang in train_languages]
        unseen_languages = [lang for lang in test_languages if lang not in train_languages]
        
        print(f"\n📊 Test Configuration:")
        print(f"  Total samples: {len(test_dataset)}")
        print(f"  Languages: {test_languages}")
        print(f"  Seen during training: {seen_languages}")
        print(f"  UNSEEN (zero-shot): {unseen_languages}")
        
        # Evaluate
        results = self.evaluate_dataset(
            test_loader,
            dataset_name="Zero-Shot Test",
            return_predictions=True,
            analyze_routing=True
        )
        
        # Separate results for seen vs unseen
        seen_metrics = {}
        unseen_metrics = {}
        
        for lang, metrics in results['per_language'].items():
            if lang in seen_languages:
                seen_metrics[lang] = metrics
            else:
                unseen_metrics[lang] = metrics
        
        # Compute aggregated metrics
        if seen_metrics:
            seen_avg = {
                'accuracy': np.mean([m['accuracy'] for m in seen_metrics.values()]),
                'f1_macro': np.mean([m['f1_macro'] for m in seen_metrics.values()])
            }
        else:
            seen_avg = {'accuracy': 0.0, 'f1_macro': 0.0}
        
        if unseen_metrics:
            unseen_avg = {
                'accuracy': np.mean([m['accuracy'] for m in unseen_metrics.values()]),
                'f1_macro': np.mean([m['f1_macro'] for m in unseen_metrics.values()])
            }
        else:
            unseen_avg = {'accuracy': 0.0, 'f1_macro': 0.0}
        
        # Store results
        self.results = {
            'overall': results['overall'],
            'seen_languages': seen_metrics,
            'unseen_languages': unseen_metrics,
            'seen_average': seen_avg,
            'unseen_average': unseen_avg,
            'routing_analysis': results['routing_analysis'],
            'raw_predictions': {
                'predictions': results['predictions'],
                'labels': results['labels'],
                'languages': results['languages'],
                'confidences': results['confidences']
            }
        }
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        # Overall metrics
        print(f"\n📈 Overall Performance:")
        print(f"  Accuracy: {self.results['overall']['accuracy']:.4f}")
        print(f"  F1 (macro): {self.results['overall']['f1_macro']:.4f}")
        print(f"  F1 (weighted): {self.results['overall']['f1_weighted']:.4f}")
        
        # Seen vs Unseen comparison
        print(f"\n🔍 Transfer Analysis:")
        print(f"  Seen languages (avg): {self.results['seen_average']['accuracy']:.4f}")
        print(f"  Unseen languages (avg): {self.results['unseen_average']['accuracy']:.4f}")
        gap = self.results['seen_average']['accuracy'] - self.results['unseen_average']['accuracy']
        print(f"  Transfer gap: {gap:.4f}")
        
        # Per-language results
        print(f"\n📊 Per-Language Results:")
        
        if self.results['seen_languages']:
            print("\n  Seen Languages:")
            for lang, metrics in sorted(self.results['seen_languages'].items()):
                print(f"    {lang}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
        
        if self.results['unseen_languages']:
            print("\n  Unseen Languages (Zero-Shot):")
            for lang, metrics in sorted(self.results['unseen_languages'].items()):
                print(f"    {lang}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
        
        # Routing analysis
        if self.results['routing_analysis']:
            print(f"\n🔀 Routing Analysis:")
            routing = self.results['routing_analysis']
            print(f"  Number of adapters: {routing['n_adapters']}")
            
            print("\n  Routing entropy (diversity):")
            for lang, entropy in sorted(routing['routing_entropy'].items()):
                print(f"    {lang}: {entropy:.3f}")


# ============================================================================
# VISUALIZATION
# ============================================================================

class ZeroShotVisualizer:
    """Visualization tools for zero-shot evaluation results."""
    
    @staticmethod
    def plot_language_performance(
        results: Dict,
        save_path: Optional[str] = None
    ):
        """Plot performance across languages."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data
        seen_langs = list(results['seen_languages'].keys())
        seen_accs = [results['seen_languages'][l]['accuracy'] for l in seen_langs]
        
        unseen_langs = list(results['unseen_languages'].keys())
        unseen_accs = [results['unseen_languages'][l]['accuracy'] for l in unseen_langs]
        
        # Plot seen languages
        if seen_langs:
            axes[0].bar(seen_langs, seen_accs, color='steelblue')
            axes[0].set_title('Seen Languages (In Training)', fontsize=14)
            axes[0].set_ylabel('Accuracy', fontsize=12)
            axes[0].set_ylim([0, 1])
            axes[0].axhline(y=np.mean(seen_accs), color='red', linestyle='--', 
                          label=f'Avg: {np.mean(seen_accs):.3f}')
            axes[0].legend()
        
        # Plot unseen languages
        if unseen_langs:
            axes[1].bar(unseen_langs, unseen_accs, color='coral')
            axes[1].set_title('Unseen Languages (Zero-Shot)', fontsize=14)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].set_ylim([0, 1])
            axes[1].axhline(y=np.mean(unseen_accs), color='red', linestyle='--',
                          label=f'Avg: {np.mean(unseen_accs):.3f}')
            axes[1].legend()
        
        plt.suptitle('Zero-Shot Cross-Lingual Transfer Performance', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Plot saved: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_routing_heatmap(
        routing_analysis: Dict,
        save_path: Optional[str] = None
    ):
        """Plot adapter routing patterns as heatmap."""
        if not routing_analysis:
            print("No routing analysis available")
            return
        
        # Prepare data
        languages = sorted(routing_analysis['average_routing'].keys())
        n_adapters = routing_analysis['n_adapters']
        
        routing_matrix = np.array([
            routing_analysis['average_routing'][lang]
            for lang in languages
        ])
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            routing_matrix,
            xticklabels=[f'A{i}' for i in range(n_adapters)],
            yticklabels=languages,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Routing Weight'}
        )
        
        plt.title('Average Adapter Routing Weights by Language', fontsize=14)
        plt.xlabel('Adapter', fontsize=12)
        plt.ylabel('Language', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Heatmap saved: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names: List[str] = None,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix."""
        if class_names is None:
            class_names = ['Class 0', 'Class 1', 'Class 2']
        
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title('Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Confusion matrix saved: {save_path}")
        
        plt.show()


# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

class ComparativeAnalyzer:
    """Compare TADR with baseline models."""
    
    @staticmethod
    def compare_with_baseline(
        tadr_results: Dict,
        baseline_results: Dict,
        model_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare TADR with baseline results.
        
        Args:
            tadr_results: TADR evaluation results
            baseline_results: Baseline evaluation results
            model_names: Names for the models
        
        Returns:
            Comparison DataFrame
        """
        if model_names is None:
            model_names = ['TADR', 'Baseline']
        
        comparison_data = []
        
        # Overall comparison
        comparison_data.append({
            'Model': model_names[0],
            'Metric': 'Overall Accuracy',
            'Value': tadr_results['overall']['accuracy']
        })
        comparison_data.append({
            'Model': model_names[1],
            'Metric': 'Overall Accuracy',
            'Value': baseline_results['overall']['accuracy']
        })
        
        # Zero-shot comparison
        if 'unseen_average' in tadr_results:
            comparison_data.append({
                'Model': model_names[0],
                'Metric': 'Zero-Shot Accuracy',
                'Value': tadr_results['unseen_average']['accuracy']
            })
        
        if 'unseen_average' in baseline_results:
            comparison_data.append({
                'Model': model_names[1],
                'Metric': 'Zero-Shot Accuracy',
                'Value': baseline_results['unseen_average']['accuracy']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Pivot for better readability
        comparison_table = df.pivot(index='Metric', columns='Model', values='Value')
        
        # Calculate improvements
        if len(model_names) == 2:
            comparison_table['Improvement'] = (
                comparison_table[model_names[0]] - comparison_table[model_names[1]]
            )
            comparison_table['Improvement %'] = (
                comparison_table['Improvement'] / comparison_table[model_names[1]] * 100
            )
        
        return comparison_table


# ============================================================================
# RESULT EXPORTER
# ============================================================================

class ResultExporter:
    """Export evaluation results in various formats."""
    
    @staticmethod
    def export_results(
        results: Dict,
        output_dir: str,
        experiment_name: str = None
    ):
        """
        Export results to multiple formats.
        
        Args:
            results: Evaluation results
            output_dir: Output directory
            experiment_name: Name for the experiment
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"zero_shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save as JSON
        json_path = os.path.join(output_dir, f'{experiment_name}_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = ResultExporter._prepare_for_json(results)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"  Results saved: {json_path}")
        
        # Save as CSV (summary)
        csv_path = os.path.join(output_dir, f'{experiment_name}_summary.csv')
        summary_df = ResultExporter._create_summary_df(results)
        summary_df.to_csv(csv_path, index=False)
        print(f"  Summary saved: {csv_path}")
        
        # Save as pickle (complete)
        pkl_path = os.path.join(output_dir, f'{experiment_name}_complete.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"  Complete results saved: {pkl_path}")
        
        # Create detailed report
        report_path = os.path.join(output_dir, f'{experiment_name}_report.txt')
        ResultExporter._create_report(results, report_path)
        print(f"  Report saved: {report_path}")
    
    @staticmethod
    def _prepare_for_json(obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {k: ResultExporter._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ResultExporter._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    @staticmethod
    def _create_summary_df(results: Dict) -> pd.DataFrame:
        """Create summary DataFrame from results."""
        summary_data = []
        
        # Add overall metrics
        summary_data.append({
            'Language': 'OVERALL',
            'Type': 'All',
            'Accuracy': results['overall']['accuracy'],
            'F1_Macro': results['overall']['f1_macro'],
            'F1_Weighted': results['overall']['f1_weighted'],
            'Samples': results['overall']['n_samples']
        })
        
        # Add per-language metrics
        for lang, metrics in results.get('seen_languages', {}).items():
            summary_data.append({
                'Language': lang,
                'Type': 'Seen',
                'Accuracy': metrics['accuracy'],
                'F1_Macro': metrics['f1_macro'],
                'F1_Weighted': metrics['f1_weighted'],
                'Samples': metrics['n_samples']
            })
        
        for lang, metrics in results.get('unseen_languages', {}).items():
            summary_data.append({
                'Language': lang,
                'Type': 'Unseen',
                'Accuracy': metrics['accuracy'],
                'F1_Macro': metrics['f1_macro'],
                'F1_Weighted': metrics['f1_weighted'],
                'Samples': metrics['n_samples']
            })
        
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def _create_report(results: Dict, report_path: str):
        """Create detailed text report."""
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ZERO-SHOT CROSS-LINGUAL TRANSFER EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Timestamp
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall results
            f.write("OVERALL PERFORMANCE\n")
            f.write("-"*40 + "\n")
            f.write(f"Accuracy: {results['overall']['accuracy']:.4f}\n")
            f.write(f"F1 (macro): {results['overall']['f1_macro']:.4f}\n")
            f.write(f"F1 (weighted): {results['overall']['f1_weighted']:.4f}\n")
            f.write(f"Total samples: {results['overall']['n_samples']}\n\n")
            
            # Transfer analysis
            if 'seen_average' in results:
                f.write("TRANSFER ANALYSIS\n")
                f.write("-"*40 + "\n")
                f.write(f"Seen languages (avg): {results['seen_average']['accuracy']:.4f}\n")
                f.write(f"Unseen languages (avg): {results['unseen_average']['accuracy']:.4f}\n")
                gap = results['seen_average']['accuracy'] - results['unseen_average']['accuracy']
                f.write(f"Transfer gap: {gap:.4f}\n\n")
            
            # Per-language results
            f.write("PER-LANGUAGE RESULTS\n")
            f.write("-"*40 + "\n")
            
            if results.get('seen_languages'):
                f.write("\nSeen Languages:\n")
                for lang, metrics in sorted(results['seen_languages'].items()):
                    f.write(f"  {lang:10s} - Acc: {metrics['accuracy']:.4f}, "
                           f"F1: {metrics['f1_macro']:.4f}, "
                           f"Samples: {metrics['n_samples']}\n")
            
            if results.get('unseen_languages'):
                f.write("\nUnseen Languages (Zero-Shot):\n")
                for lang, metrics in sorted(results['unseen_languages'].items()):
                    f.write(f"  {lang:10s} - Acc: {metrics['accuracy']:.4f}, "
                           f"F1: {metrics['f1_macro']:.4f}, "
                           f"Samples: {metrics['n_samples']}\n")
            
            # Routing analysis
            if results.get('routing_analysis'):
                f.write("\n\nROUTING ANALYSIS\n")
                f.write("-"*40 + "\n")
                routing = results['routing_analysis']
                f.write(f"Number of adapters: {routing['n_adapters']}\n\n")
                
                f.write("Routing Entropy (diversity):\n")
                for lang, entropy in sorted(routing['routing_entropy'].items()):
                    f.write(f"  {lang}: {entropy:.3f}\n")
                
                f.write("\nTop Adapters per Language:\n")
                for lang, top in sorted(routing['top_adapters'].items()):
                    f.write(f"  {lang}: {top['indices']} "
                           f"(weights: {[f'{w:.3f}' for w in top['weights']]})\n")


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def load_trained_model(
    checkpoint_path: str,
    model_config: Dict,
    device: str = 'cuda'
) -> CompleteTADRModel:
    """
    Load trained TADR model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_config: Model configuration
        device: Device to load to
    
    Returns:
        Loaded model
    """
    print(f"\n📂 Loading model from: {checkpoint_path}")
    
    # Create model
    model = create_tadr_model(**model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best val acc: {checkpoint.get('best_val_acc', 'N/A'):.4f}")
    
    return model


def run_zero_shot_evaluation(
    checkpoint_path: str,
    data_dir: str = './data_zeroshot',
    output_dir: str = './zero_shot_results',
    model_name: str = 'xlm-roberta-base',
    batch_size: int = 32,
    device: str = None,
    visualize: bool = True,
    export_results: bool = True
):
    """
    Complete zero-shot evaluation pipeline.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_dir: Directory with test data
        output_dir: Output directory for results
        model_name: Base model name
        batch_size: Batch size for evaluation
        device: Device to use
        visualize: Whether to create visualizations
        export_results: Whether to export results
    """

    XNLI_LANG_MAPPING = {
        'ar': 'arz', 'bg': 'bul', 'de': 'deu', 'el': 'ell',
        'en': 'eng', 'es': 'spa', 'fr': 'fra', 'hi': 'hin',
        'ru': 'rus', 'sw': 'swh', 'th': 'tha', 'tr': 'tur',
        'ur': 'urd', 'vi': 'vie', 'zh': 'cmn'
    }
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "🚀"*35)
    print("ZERO-SHOT TRANSFER EVALUATION PIPELINE")
    print("🚀"*35 + "\n")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print("Step 1: Loading test data...")
    train_data, val_data, test_data = load_preprocessed_data(data_dir)
    
    # Load configuration to get training languages
    config_path = os.path.join(data_dir, 'config.json')
    with open(config_path, 'r') as f:
        data_config = json.load(f)
    
    train_languages = [XNLI_LANG_MAPPING.get(lang, lang) 
                      for lang in data_config['train_languages']]
    
    print(f"  Training languages: {train_languages}")
    print(f"  Test samples: {len(test_data['texts'])}")
    
    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    
    print("\nStep 2: Loading trained model...")
    
    model_config = {
        'model_name': model_name,
        'feature_file': 'wals_features.csv',
        'num_adapters': 8,
        'adapter_bottleneck': 48,
        'num_classes': 3,
        'gating_type': 'softmax',
        'num_adapter_layers': 4,
        'device': device
    }
    
    model = load_trained_model(checkpoint_path, model_config, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # ========================================================================
    # EVALUATE
    # ========================================================================
    
    print("\nStep 3: Running zero-shot evaluation...")
    
    evaluator = ZeroShotEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        verbose=True
    )
    
    results = evaluator.evaluate_zero_shot_transfer(
        test_data=test_data,
        train_languages=train_languages,
        batch_size=batch_size
    )
    
    # ========================================================================
    # VISUALIZE
    # ========================================================================
    
    if visualize:
        print("\nStep 4: Creating visualizations...")
        os.makedirs(output_dir, exist_ok=True)
        
        visualizer = ZeroShotVisualizer()
        
        # Performance bar plot
        visualizer.plot_language_performance(
            results,
            save_path=os.path.join(output_dir, 'language_performance.png')
        )
        
        # Routing heatmap
        if results.get('routing_analysis'):
            visualizer.plot_routing_heatmap(
                results['routing_analysis'],
                save_path=os.path.join(output_dir, 'routing_heatmap.png')
            )
        
        # Confusion matrix for overall predictions
        if results.get('raw_predictions'):
            visualizer.plot_confusion_matrix(
                np.array(results['raw_predictions']['predictions']),
                np.array(results['raw_predictions']['labels']),
                class_names=['Entailment', 'Neutral', 'Contradiction'],
                save_path=os.path.join(output_dir, 'confusion_matrix.png')
            )
    
    # ========================================================================
    # EXPORT RESULTS
    # ========================================================================
    
    if export_results:
        print("\nStep 5: Exporting results...")
        
        exporter = ResultExporter()
        exporter.export_results(
            results,
            output_dir=output_dir,
            experiment_name='tadr_zero_shot'
        )
    
    print("\n" + "="*70)
    print("✅ ZERO-SHOT EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    
    return results


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TADR Zero-Shot Cross-Lingual Transfer Evaluation'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./tadr_checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data_zeroshot',
        help='Directory with test data'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./zero_shot_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='xlm-roberta-base',
        help='Base model name'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)'
    )
    
    parser.add_argument(
        '--no_visualize',
        action='store_true',
        help='Skip visualization'
    )
    
    parser.add_argument(
        '--no_export',
        action='store_true',
        help='Skip result export'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_zero_shot_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        visualize=not args.no_visualize,
        export_results=not args.no_export
    )
    
    return results


if __name__ == "__main__":
    # You can either use the CLI or call directly
    if len(os.sys.argv) > 1:
        results = main()
    else:
        # Direct call with default parameters
        results = run_zero_shot_evaluation(
            checkpoint_path='./tadr_checkpoints/best_model.pt',
            data_dir='./data_zeroshot',
            output_dir='./zero_shot_results',
            model_name='xlm-roberta-base',
            batch_size=32,
            device=None,
            visualize=True,
            export_results=True
        )