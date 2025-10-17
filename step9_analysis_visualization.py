"""
TADR Framework - Step 9: Analysis & Visualization
Comprehensive analysis tools and paper-ready visualizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


class TADRAnalyzer:
    """
    Comprehensive analysis toolkit for TADR model.
    Generates paper-ready visualizations and statistical analyses.
    """
    
    def __init__(
        self,
        model: nn.Module,
        typology_module: nn.Module,
        output_dir: str = './analysis_results'
    ):
        """
        Args:
            model: Trained TADR model
            typology_module: Typology feature module
            output_dir: Directory to save results
        """
        self.model = model
        self.typology_module = typology_module
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        
        self.model.eval()
    
    def analyze_routing_patterns(
        self,
        language_results: Dict[str, Dict[str, Any]],
        save_prefix: str = 'routing_analysis'
    ) -> Dict[str, Any]:
        """
        Comprehensive routing pattern analysis.
        
        Args:
            language_results: Results from ZeroShotEvaluator
            save_prefix: Prefix for saved files
        
        Returns:
            Analysis results
        """
        print("\n" + "="*70)
        print("ROUTING PATTERN ANALYSIS")
        print("="*70)
        
        analysis = {}
        
        # 1. Adapter usage distribution
        print("\n1. Analyzing adapter usage distribution...")
        analysis['adapter_usage'] = self._analyze_adapter_usage(language_results)
        
        # 2. Routing entropy analysis
        print("2. Analyzing routing entropy...")
        analysis['entropy'] = self._analyze_routing_entropy(language_results)
        
        # 3. Sparsity analysis
        print("3. Analyzing routing sparsity...")
        analysis['sparsity'] = self._analyze_sparsity(language_results)
        
        # 4. Language clustering by routing
        print("4. Clustering languages by routing patterns...")
        analysis['clustering'] = self._cluster_by_routing(language_results)
        
        # Save analysis
        self._save_json(analysis, f'{save_prefix}_results.json')
        
        return analysis
    
    def _analyze_adapter_usage(
        self,
        language_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze which adapters are used most frequently."""
        all_usage = defaultdict(int)
        per_lang_usage = {}
        
        for lang_id, result in language_results.items():
            usage = result['routing_stats']['adapter_usage']
            per_lang_usage[lang_id] = usage
            
            for adapter_id, count in usage.items():
                all_usage[adapter_id] += count
        
        # Compute statistics
        total_selections = sum(all_usage.values())
        usage_distribution = {
            k: v / total_selections for k, v in all_usage.items()
        }
        
        # Find most/least used adapters
        sorted_usage = sorted(usage_distribution.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'overall_distribution': dict(usage_distribution),
            'per_language': per_lang_usage,
            'most_used': sorted_usage[:3],
            'least_used': sorted_usage[-3:],
            'gini_coefficient': self._compute_gini(list(usage_distribution.values()))
        }
    
    def _analyze_routing_entropy(
        self,
        language_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze routing entropy across languages."""
        entropies = {}
        
        for lang_id, result in language_results.items():
            entropies[lang_id] = result['routing_stats']['avg_entropy']
        
        return {
            'per_language': entropies,
            'mean': float(np.mean(list(entropies.values()))),
            'std': float(np.std(list(entropies.values()))),
            'min': float(np.min(list(entropies.values()))),
            'max': float(np.max(list(entropies.values())))
        }
    
    def _analyze_sparsity(
        self,
        language_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze routing sparsity."""
        sparsities = {}
        
        for lang_id, result in language_results.items():
            sparsities[lang_id] = result['routing_stats']['avg_sparsity']
        
        return {
            'per_language': sparsities,
            'mean': float(np.mean(list(sparsities.values()))),
            'std': float(np.std(list(sparsities.values())))
        }
    
    def _cluster_by_routing(
        self,
        language_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Cluster languages based on routing patterns."""
        # Extract routing weight vectors
        languages = list(language_results.keys())
        routing_vectors = np.array([
            language_results[lang]['routing_stats']['avg_weights']
            for lang in languages
        ])
        
        # Compute pairwise similarities
        similarities = np.dot(routing_vectors, routing_vectors.T)
        
        return {
            'similarity_matrix': similarities.tolist(),
            'languages': languages
        }
    
    def _compute_gini(self, values: List[float]) -> float:
        """Compute Gini coefficient for inequality measurement."""
        sorted_values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * sum((i + 1) * v for i, v in enumerate(sorted_values))) / (n * sum(sorted_values)) - (n + 1) / n
    
    def analyze_typological_patterns(
        self,
        languages: List[str],
        save_prefix: str = 'typology_analysis'
    ) -> Dict[str, Any]:
        """
        Analyze typological patterns and embeddings.
        
        Args:
            languages: List of language IDs
            save_prefix: Prefix for saved files
        
        Returns:
            Analysis results
        """
        print("\n" + "="*70)
        print("TYPOLOGICAL PATTERN ANALYSIS")
        print("="*70)
        
        # Get typology embeddings
        embeddings = []
        for lang in languages:
            emb = self.typology_module.get_embedding(lang).detach().cpu().numpy()
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Compute pairwise similarities
        print("\n1. Computing pairwise typological similarities...")
        similarities = self._compute_pairwise_similarities(embeddings)
        
        # Dimensionality reduction
        print("2. Performing dimensionality reduction...")
        reduced_embeddings = self._reduce_dimensions(embeddings)
        
        analysis = {
            'embeddings': embeddings.tolist(),
            'similarities': similarities.tolist(),
            'reduced_embeddings': {
                'pca': reduced_embeddings['pca'].tolist(),
                'tsne': reduced_embeddings['tsne'].tolist()
            },
            'languages': languages
        }
        
        # Save analysis
        self._save_json(analysis, f'{save_prefix}_results.json')
        
        return analysis
    
    def _compute_pairwise_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        similarities = np.dot(normalized, normalized.T)
        return similarities
    
    def _reduce_dimensions(
        self,
        embeddings: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Reduce embeddings to 2D for visualization."""
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        tsne_result = tsne.fit_transform(embeddings)
        
        return {
            'pca': pca_result,
            'tsne': tsne_result
        }
    
    def analyze_performance_factors(
        self,
        results: Dict[str, Dict[str, Any]],
        typology_similarities: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze factors affecting zero-shot performance.
        
        Args:
            results: Evaluation results
            typology_similarities: Optional pre-computed similarities
        
        Returns:
            Analysis results
        """
        print("\n" + "="*70)
        print("PERFORMANCE FACTOR ANALYSIS")
        print("="*70)
        
        languages = list(results.keys())
        accuracies = [results[lang]['accuracy'] for lang in languages]
        
        # 1. Routing characteristics vs performance
        print("\n1. Analyzing routing characteristics vs performance...")
        entropies = [results[lang]['routing_stats']['avg_entropy'] for lang in languages]
        sparsities = [results[lang]['routing_stats']['avg_sparsity'] for lang in languages]
        
        entropy_corr, _ = pearsonr(accuracies, entropies)
        sparsity_corr, _ = pearsonr(accuracies, sparsities)
        
        print(f"   Entropy-Accuracy correlation: {entropy_corr:.4f}")
        print(f"   Sparsity-Accuracy correlation: {sparsity_corr:.4f}")
        
        # 2. Dataset size vs performance
        print("\n2. Analyzing dataset size effect...")
        sizes = [results[lang]['num_samples'] for lang in languages]
        size_corr, _ = pearsonr(accuracies, sizes)
        print(f"   Dataset size-Accuracy correlation: {size_corr:.4f}")
        
        analysis = {
            'correlations': {
                'entropy_accuracy': float(entropy_corr),
                'sparsity_accuracy': float(sparsity_corr),
                'size_accuracy': float(size_corr)
            },
            'per_language': {
                lang: {
                    'accuracy': results[lang]['accuracy'],
                    'entropy': results[lang]['routing_stats']['avg_entropy'],
                    'sparsity': results[lang]['routing_stats']['avg_sparsity'],
                    'size': results[lang]['num_samples']
                }
                for lang in languages
            }
        }
        
        return analysis
    
    # ========================================================================
    # VISUALIZATION METHODS
    # ========================================================================
    
    def plot_routing_heatmap(
        self,
        language_results: Dict[str, Dict[str, Any]],
        figsize: Tuple[int, int] = (12, 8),
        save_name: str = 'routing_heatmap.png'
    ):
        """
        Create heatmap of routing patterns across languages.
        
        Args:
            language_results: Results from evaluation
            figsize: Figure size
            save_name: Filename to save
        """
        print("\nGenerating routing heatmap...")
        
        languages = list(language_results.keys())
        num_adapters = len(language_results[languages[0]]['routing_stats']['avg_weights'])
        
        # Create matrix
        matrix = np.array([
            language_results[lang]['routing_stats']['avg_weights']
            for lang in languages
        ])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(num_adapters))
        ax.set_yticks(np.arange(len(languages)))
        ax.set_xticklabels([f'A{i}' for i in range(num_adapters)])
        ax.set_yticklabels(languages)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Routing Weight', rotation=270, labelpad=20)
        
        # Add values
        for i in range(len(languages)):
            for j in range(num_adapters):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Routing Patterns Across Languages', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Adapters', fontsize=12)
        ax.set_ylabel('Languages', fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'figures', save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_performance_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        baseline_results: Optional[Dict[str, Dict[str, Any]]] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_name: str = 'performance_comparison.png'
    ):
        """
        Plot performance comparison across languages.
        
        Args:
            results: TADR results
            baseline_results: Optional baseline results
            figsize: Figure size
            save_name: Filename to save
        """
        print("\nGenerating performance comparison plot...")
        
        languages = list(results.keys())
        tadr_accs = [results[lang]['accuracy'] for lang in languages]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(languages))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, tadr_accs, width, label='TADR', color='#2E86AB')
        
        if baseline_results:
            baseline_accs = [baseline_results[lang]['accuracy'] for lang in languages]
            bars2 = ax.bar(x + width/2, baseline_accs, width, label='Baseline', color='#A23B72')
        
        ax.set_xlabel('Languages', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Zero-Shot Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(languages, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'figures', save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_typology_space(
        self,
        languages: List[str],
        results: Optional[Dict[str, Dict[str, Any]]] = None,
        method: str = 'tsne',
        figsize: Tuple[int, int] = (10, 8),
        save_name: str = 'typology_space.png'
    ):
        """
        Visualize typological space in 2D.
        
        Args:
            languages: List of language IDs
            results: Optional results to color by performance
            method: 'tsne' or 'pca'
            figsize: Figure size
            save_name: Filename to save
        """
        print(f"\nGenerating typology space visualization ({method.upper()})...")
        
        # Get embeddings
        embeddings = []
        for lang in languages:
            emb = self.typology_module.get_embedding(lang).detach().cpu().numpy()
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Reduce dimensions
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(languages)-1))
        else:
            reducer = PCA(n_components=2)
        
        reduced = reducer.fit_transform(embeddings)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color by performance if results provided
        if results:
            colors = [results[lang]['accuracy'] for lang in languages]
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=colors, 
                               cmap='RdYlGn', s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Accuracy', rotation=270, labelpad=20)
        else:
            ax.scatter(reduced[:, 0], reduced[:, 1], s=200, alpha=0.7, 
                      edgecolors='black', linewidth=1.5)
        
        # Add labels
        for i, lang in enumerate(languages):
            ax.annotate(lang, (reduced[i, 0], reduced[i, 1]), 
                       fontsize=10, fontweight='bold',
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12, fontweight='bold')
        ax.set_title('Typological Space Visualization', fontsize=14, fontweight='bold', pad=20)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'figures', save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_similarity_performance_correlation(
        self,
        correlations: Dict[str, Any],
        figsize: Tuple[int, int] = (10, 6),
        save_name: str = 'similarity_correlation.png'
    ):
        """
        Plot correlation between typological similarity and performance.
        
        Args:
            correlations: Correlation data from analyze_typological_transfer
            figsize: Figure size
            save_name: Filename to save
        """
        print("\nGenerating similarity-performance correlation plot...")
        
        languages = list(correlations.keys())
        accuracies = [correlations[lang]['accuracy'] for lang in languages]
        similarities = [correlations[lang]['max_similarity'] for lang in languages]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        ax.scatter(similarities, accuracies, s=150, alpha=0.6, 
                  edgecolors='black', linewidth=1.5, color='#2E86AB')
        
        # Add language labels
        for i, lang in enumerate(languages):
            ax.annotate(lang, (similarities[i], accuracies[i]),
                       fontsize=9, xytext=(5, 5), textcoords='offset points')
        
        # Fit line
        z = np.polyfit(similarities, accuracies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(similarities), max(similarities), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
        
        # Compute correlation
        corr, p_value = pearsonr(similarities, accuracies)
        
        ax.set_xlabel('Max Typological Similarity to Training Languages', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Zero-Shot Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'Typological Similarity vs Performance\n(r={corr:.3f}, p={p_value:.4f})', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'figures', save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_routing_entropy_distribution(
        self,
        results: Dict[str, Dict[str, Any]],
        figsize: Tuple[int, int] = (10, 6),
        save_name: str = 'entropy_distribution.png'
    ):
        """
        Plot distribution of routing entropy across languages.
        
        Args:
            results: Evaluation results
            figsize: Figure size
            save_name: Filename to save
        """
        print("\nGenerating routing entropy distribution plot...")
        
        languages = list(results.keys())
        entropies = [results[lang]['routing_stats']['avg_entropy'] for lang in languages]
        accuracies = [results[lang]['accuracy'] for lang in languages]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(entropies, bins=15, color='#A23B72', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(entropies), color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.set_xlabel('Routing Entropy', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Distribution of Routing Entropy', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Scatter: entropy vs accuracy
        ax2.scatter(entropies, accuracies, s=150, alpha=0.6, 
                   edgecolors='black', linewidth=1.5, color='#2E86AB')
        
        for i, lang in enumerate(languages):
            ax2.annotate(lang, (entropies[i], accuracies[i]),
                        fontsize=8, xytext=(3, 3), textcoords='offset points')
        
        # Correlation
        corr, _ = pearsonr(entropies, accuracies)
        ax2.set_xlabel('Routing Entropy', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax2.set_title(f'Entropy vs Accuracy (r={corr:.3f})', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'figures', save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_adapter_usage_pie(
        self,
        adapter_usage: Dict[int, int],
        figsize: Tuple[int, int] = (8, 8),
        save_name: str = 'adapter_usage.png'
    ):
        """
        Plot pie chart of adapter usage.
        
        Args:
            adapter_usage: Adapter usage counts
            figsize: Figure size
            save_name: Filename to save
        """
        print("\nGenerating adapter usage pie chart...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = [f'Adapter {k}' for k in adapter_usage.keys()]
        sizes = list(adapter_usage.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax.set_title('Adapter Usage Distribution', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'figures', save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def generate_full_report(
        self,
        train_langs: List[str],
        test_langs: List[str],
        results: Dict[str, Dict[str, Any]],
        baseline_results: Optional[Dict[str, Dict[str, Any]]] = None,
        typology_correlations: Optional[Dict[str, Any]] = None
    ):
        """
        Generate a comprehensive analysis report with all visualizations.
        
        Args:
            train_langs: Training languages
            test_langs: Test (zero-shot) languages
            results: Evaluation results
            baseline_results: Optional baseline results
            typology_correlations: Optional typology correlation data
        """
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("="*70)
        
        # 1. Routing heatmap
        self.plot_routing_heatmap(results)
        
        # 2. Performance comparison
        self.plot_performance_comparison(results, baseline_results)
        
        # 3. Typology space
        all_langs = train_langs + test_langs
        self.plot_typology_space(all_langs, results, method='tsne')
        self.plot_typology_space(all_langs, results, method='pca', 
                                save_name='typology_space_pca.png')
        
        # 4. Entropy distribution
        self.plot_routing_entropy_distribution(results)
        
        # 5. Adapter usage
        all_usage = defaultdict(int)
        for lang, res in results.items():
            for adapter_id, count in res['routing_stats']['adapter_usage'].items():
                all_usage[adapter_id] += count
        self.plot_adapter_usage_pie(dict(all_usage))
        
        # 6. Similarity correlation (if available)
        if typology_correlations:
            self.plot_similarity_performance_correlation(typology_correlations)
        
        print("\n✅ Full report generated successfully!")
        print(f"All figures saved to {os.path.join(self.output_dir, 'figures')}")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _save_json(self, data: Dict[str, Any], filename: str):
        """Save data to JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved analysis to {filepath}")
    
    def create_latex_table(
        self,
        results: Dict[str, Dict[str, Any]],
        baseline_results: Optional[Dict[str, Dict[str, Any]]] = None,
        save_name: str = 'results_table.tex'
    ) -> str:
        """
        Create LaTeX table for paper.
        
        Args:
            results: TADR results
            baseline_results: Optional baseline results
            save_name: Filename to save
        
        Returns:
            LaTeX table string
        """
        print("\nGenerating LaTeX table...")
        
        languages = sorted(results.keys())
        
        # Start table
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Zero-Shot Transfer Performance}\n"
        latex += "\\label{tab:zero_shot_results}\n"
        
        if baseline_results:
            latex += "\\begin{tabular}{l|cc|c}\n"
            latex += "\\hline\n"
            latex += "Language & TADR & Baseline & Improvement \\\\\n"
            latex += "\\hline\n"
            
            for lang in languages:
                tadr_acc = results[lang]['accuracy']
                base_acc = baseline_results[lang]['accuracy']
                improvement = ((tadr_acc - base_acc) / base_acc) * 100
                
                latex += f"{lang} & {tadr_acc:.3f} & {base_acc:.3f} & {improvement:+.1f}\\% \\\\\n"
            
            # Average
            avg_tadr = np.mean([results[l]['accuracy'] for l in languages])
            avg_base = np.mean([baseline_results[l]['accuracy'] for l in languages])
            avg_imp = ((avg_tadr - avg_base) / avg_base) * 100
            
            latex += "\\hline\n"
            latex += f"Average & {avg_tadr:.3f} & {avg_base:.3f} & {avg_imp:+.1f}\\% \\\\\n"
        else:
            latex += "\\begin{tabular}{l|c|c|c}\n"
            latex += "\\hline\n"
            latex += "Language & Accuracy & Entropy & Sparsity \\\\\n"
            latex += "\\hline\n"
            
            for lang in languages:
                acc = results[lang]['accuracy']
                entropy = results[lang]['routing_stats']['avg_entropy']
                sparsity = results[lang]['routing_stats']['avg_sparsity']
                
                latex += f"{lang} & {acc:.3f} & {entropy:.3f} & {sparsity:.3f} \\\\\n"
            
            # Average
            avg_acc = np.mean([results[l]['accuracy'] for l in languages])
            avg_entropy = np.mean([results[l]['routing_stats']['avg_entropy'] for l in languages])
            avg_sparsity = np.mean([results[l]['routing_stats']['avg_sparsity'] for l in languages])
            
            latex += "\\hline\n"
            latex += f"Average & {avg_acc:.3f} & {avg_entropy:.3f} & {avg_sparsity:.3f} \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        # Save to file
        filepath = os.path.join(self.output_dir, save_name)
        with open(filepath, 'w') as f:
            f.write(latex)
        
        print(f"LaTeX table saved to {filepath}")
        return latex
    
    def analyze_typological_transfer(
        self,
        train_langs: List[str],
        test_langs: List[str],
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze transfer based on typological similarities.
        
        Args:
            train_langs: Training languages
            test_langs: Test languages
            results: Evaluation results
        
        Returns:
            Transfer analysis results
        """
        print("\n" + "="*70)
        print("TYPOLOGICAL TRANSFER ANALYSIS")
        print("="*70)
        
        # Get typology embeddings
        train_embeddings = []
        test_embeddings = []
        
        for lang in train_langs:
            emb = self.typology_module.get_embedding(lang).detach().cpu().numpy()
            train_embeddings.append(emb)
        
        for lang in test_langs:
            emb = self.typology_module.get_embedding(lang).detach().cpu().numpy()
            test_embeddings.append(emb)
        
        train_embeddings = np.array(train_embeddings)
        test_embeddings = np.array(test_embeddings)
        
        # Compute similarities
        similarities = {}
        for i, test_lang in enumerate(test_langs):
            # Compute similarity to each training language
            sims = []
            for j, train_lang in enumerate(train_langs):
                sim = np.dot(test_embeddings[i], train_embeddings[j]) / (
                    np.linalg.norm(test_embeddings[i]) * np.linalg.norm(train_embeddings[j])
                )
                sims.append(sim)
            
            similarities[test_lang] = {
                'similarities': {train_langs[j]: float(sims[j]) for j in range(len(train_langs))},
                'max_similarity': float(np.max(sims)),
                'avg_similarity': float(np.mean(sims)),
                'most_similar': train_langs[np.argmax(sims)]
            }
        
        # Analyze correlation with performance
        if results:
            test_accuracies = [results[lang]['accuracy'] for lang in test_langs if lang in results]
            max_similarities = [similarities[lang]['max_similarity'] for lang in test_langs if lang in results]
            
            if len(test_accuracies) > 1:
                corr, p_value = pearsonr(max_similarities, test_accuracies)
                print(f"\nCorrelation between max similarity and accuracy: {corr:.4f} (p={p_value:.4f})")
            else:
                corr, p_value = None, None
        else:
            corr, p_value = None, None
        
        return {
            'similarities': similarities,
            'correlation': {
                'pearson_r': corr,
                'p_value': p_value
            } if corr is not None else None
        }
    
    def plot_language_family_performance(
        self,
        results: Dict[str, Dict[str, Any]],
        language_families: Dict[str, str],
        figsize: Tuple[int, int] = (12, 6),
        save_name: str = 'family_performance.png'
    ):
        """
        Plot performance grouped by language family.
        
        Args:
            results: Evaluation results
            language_families: Mapping of language to family
            figsize: Figure size
            save_name: Filename to save
        """
        print("\nGenerating language family performance plot...")
        
        # Group by family
        family_data = defaultdict(list)
        for lang, res in results.items():
            family = language_families.get(lang, 'Unknown')
            family_data[family].append(res['accuracy'])
        
        # Create box plot
        fig, ax = plt.subplots(figsize=figsize)
        
        families = list(family_data.keys())
        data = [family_data[f] for f in families]
        
        bp = ax.boxplot(data, labels=families, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(families)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel('Language Family', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Performance by Language Family', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add sample sizes
        for i, family in enumerate(families):
            n = len(family_data[family])
            ax.text(i + 1, ax.get_ylim()[0] + 0.02, f'n={n}', 
                   ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'figures', save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def export_results_csv(
        self,
        results: Dict[str, Dict[str, Any]],
        save_name: str = 'results.csv'
    ):
        """
        Export results to CSV for further analysis.
        
        Args:
            results: Evaluation results
            save_name: Filename to save
        """
        print("\nExporting results to CSV...")
        
        data = []
        for lang, res in results.items():
            row = {
                'language': lang,
                'accuracy': res['accuracy'],
                'num_samples': res['num_samples'],
                'avg_entropy': res['routing_stats']['avg_entropy'],
                'avg_sparsity': res['routing_stats']['avg_sparsity'],
                'effective_adapters': res['routing_stats']['effective_adapters']
            }
            
            # Add per-adapter usage
            for adapter_id, usage in res['routing_stats']['adapter_usage'].items():
                row[f'adapter_{adapter_id}_usage'] = usage
            
            data.append(row)
        
        df = pd.DataFrame(data)
        filepath = os.path.join(self.output_dir, save_name)
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")
        
        return df
    
    def plot_training_curves(
        self,
        training_history: Dict[str, List[float]],
        figsize: Tuple[int, int] = (12, 4),
        save_name: str = 'training_curves.png'
    ):
        """
        Plot training curves (loss, accuracy, etc.).
        
        Args:
            training_history: Dictionary with metric histories
            figsize: Figure size
            save_name: Filename to save
        """
        print("\nGenerating training curves...")
        
        num_metrics = len(training_history)
        fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
        
        if num_metrics == 1:
            axes = [axes]
        
        for ax, (metric_name, values) in zip(axes, training_history.items()):
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_title(f'{metric_name.replace("_", " ").title()} Over Time', 
                        fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'figures', save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()


# ============================================================================
# MAIN EXECUTION EXAMPLE
# ============================================================================

def run_comprehensive_analysis(
    model: nn.Module,
    typology_module: nn.Module,
    train_langs: List[str],
    test_langs: List[str],
    train_results: Dict[str, Dict[str, Any]],
    test_results: Dict[str, Dict[str, Any]],
    baseline_results: Optional[Dict[str, Dict[str, Any]]] = None,
    language_families: Optional[Dict[str, str]] = None,
    training_history: Optional[Dict[str, List[float]]] = None,
    output_dir: str = './analysis_results'
):
    """
    Run comprehensive analysis and generate all visualizations.
    
    Args:
        model: Trained TADR model
        typology_module: Typology feature module
        train_langs: Training languages
        test_langs: Test languages
        train_results: Training language results
        test_results: Test language results
        baseline_results: Optional baseline results
        language_families: Optional language family mappings
        training_history: Optional training history
        output_dir: Output directory
    
    Returns:
        Complete analysis results
    """
    print("\n" + "="*80)
    print(" TADR COMPREHENSIVE ANALYSIS SUITE ".center(80, "="))
    print("="*80)
    
    # Initialize analyzer
    analyzer = TADRAnalyzer(model, typology_module, output_dir)
    
    all_results = {**train_results, **test_results}
    
    # 1. Routing pattern analysis
    routing_analysis = analyzer.analyze_routing_patterns(
        test_results,
        save_prefix='routing_analysis'
    )
    
    # 2. Typological pattern analysis
    all_langs = train_langs + test_langs
    typology_analysis = analyzer.analyze_typological_patterns(
        all_langs,
        save_prefix='typology_analysis'
    )
    
    # 3. Performance factor analysis
    performance_analysis = analyzer.analyze_performance_factors(test_results)
    
    # 4. Typological transfer analysis
    transfer_analysis = analyzer.analyze_typological_transfer(
        train_langs,
        test_langs,
        test_results
    )
    
    # 5. Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Routing heatmap
    analyzer.plot_routing_heatmap(test_results)
    
    # Performance comparison
    analyzer.plot_performance_comparison(test_results, baseline_results)
    
    # Typology space
    analyzer.plot_typology_space(all_langs, all_results, method='tsne')
    analyzer.plot_typology_space(all_langs, all_results, method='pca', 
                                save_name='typology_space_pca.png')
    
    # Entropy distribution
    analyzer.plot_routing_entropy_distribution(test_results)
    
    # Adapter usage
    all_usage = defaultdict(int)
    for res in test_results.values():
        for adapter_id, count in res['routing_stats']['adapter_usage'].items():
            all_usage[adapter_id] += count
    analyzer.plot_adapter_usage_pie(dict(all_usage))
    
    # Similarity correlation
    if transfer_analysis['correlation']:
        correlation_data = {}
        for lang in test_langs:
            if lang in test_results:
                correlation_data[lang] = {
                    'accuracy': test_results[lang]['accuracy'],
                    'max_similarity': transfer_analysis['similarities'][lang]['max_similarity']
                }
        analyzer.plot_similarity_performance_correlation(correlation_data)
    
    # Language family performance (if available)
    if language_families:
        analyzer.plot_language_family_performance(test_results, language_families)
    
    # Training curves (if available)
    if training_history:
        analyzer.plot_training_curves(training_history)
    
    # 6. Generate tables
    print("\n" + "="*70)
    print("GENERATING TABLES")
    print("="*70)
    
    # LaTeX table
    analyzer.create_latex_table(test_results, baseline_results)
    
    # CSV export
    results_df = analyzer.export_results_csv(test_results)
    
    # 7. Create summary report
    summary = {
        'num_train_langs': len(train_langs),
        'num_test_langs': len(test_langs),
        'avg_test_accuracy': float(np.mean([r['accuracy'] for r in test_results.values()])),
        'avg_routing_entropy': float(np.mean([r['routing_stats']['avg_entropy'] 
                                             for r in test_results.values()])),
        'avg_routing_sparsity': float(np.mean([r['routing_stats']['avg_sparsity'] 
                                              for r in test_results.values()])),
        'typology_transfer_correlation': transfer_analysis['correlation'],
        'routing_analysis': routing_analysis,
        'performance_analysis': performance_analysis
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE ".center(80, "="))
    print("="*80)
    print(f"\n📊 All results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  ├── 📁 figures/           - All visualizations")
    print(f"  ├── 📄 results.csv        - Tabular results")
    print(f"  ├── 📄 results_table.tex  - LaTeX table")
    print(f"  └── 📄 analysis_summary.json - Complete analysis summary")
    
    return summary


# ============================================================================
# STATISTICAL TESTS MODULE
# ============================================================================

class StatisticalTester:
    """Additional statistical tests for rigorous analysis."""
    
    @staticmethod
    def paired_t_test(
        scores1: List[float],
        scores2: List[float]
    ) -> Tuple[float, float]:
        """
        Perform paired t-test between two sets of scores.
        
        Args:
            scores1: First set of scores
            scores2: Second set of scores
        
        Returns:
            t-statistic and p-value
        """
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        return t_stat, p_value
    
    @staticmethod
    def bootstrap_confidence_interval(
        scores: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval.
        
        Args:
            scores: List of scores
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
        
        Returns:
            (mean, lower_bound, upper_bound)
        """
        bootstrap_means = []
        n = len(scores)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_means, alpha * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
        
        return np.mean(scores), lower, upper
    
    @staticmethod
    def effect_size_cohens_d(
        scores1: List[float],
        scores2: List[float]
    ) -> float:
        """
        Compute Cohen's d effect size.
        
        Args:
            scores1: First set of scores
            scores2: Second set of scores
        
        Returns:
            Cohen's d
        """
        mean1 = np.mean(scores1)
        mean2 = np.mean(scores2)
        std1 = np.std(scores1, ddof=1)
        std2 = np.std(scores2, ddof=1)
        
        pooled_std = np.sqrt(((len(scores1) - 1) * std1**2 + 
                              (len(scores2) - 1) * std2**2) / 
                             (len(scores1) + len(scores2) - 2))
        
        return (mean1 - mean2) / pooled_std


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("TADR Analysis Suite - Ready for use!")
    print("\nExample usage:")
    print("=" * 50)
    print("""
    # Initialize analyzer
    analyzer = TADRAnalyzer(model, typology_module)
    
    # Run comprehensive analysis
    summary = run_comprehensive_analysis(
        model=model,
        typology_module=typology_module,
        train_langs=['en', 'de', 'fr'],
        test_langs=['sw', 'hi', 'ar'],
        train_results=train_results,
        test_results=test_results,
        baseline_results=baseline_results
    )
    
    # Individual analyses
    routing_analysis = analyzer.analyze_routing_patterns(test_results)
    transfer_analysis = analyzer.analyze_typological_transfer(
        train_langs, test_langs, test_results
    )
    """)