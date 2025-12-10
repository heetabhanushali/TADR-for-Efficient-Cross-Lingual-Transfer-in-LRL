"""
TADR Framework - Step 9: Analysis & Interpretability
Comprehensive analysis of typology-aware routing patterns and cross-lingual transfer
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import components from previous steps
from step5_integration import CompleteTADRModel
from step1_typology_module import TypologicalFeatureLoader
from step8_zero_shot_transfer import load_trained_model, ZeroShotEvaluator

# ============================================================================
# TYPOLOGICAL SIMILARITY ANALYZER
# ============================================================================

class TypologicalSimilarityAnalyzer:
    """
    Analyzes relationships between typological features, routing patterns,
    and cross-lingual transfer performance.
    """
    
    def __init__(
        self,
        model: CompleteTADRModel,
        feature_file: str = 'wals_features.csv',
        device: str = 'cuda'
    ):
        """
        Args:
            model: Trained TADR model
            feature_file: Path to WALS features
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        
        # Load typological features
        self.feature_loader = TypologicalFeatureLoader(feature_file)
        self.languages = self.feature_loader.get_all_languages()
        
        # Cache for embeddings and features
        self.typology_embeddings = {}
        self.typology_features = {}
        self.routing_patterns = {}
        
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Precompute typology embeddings for all languages."""
        print("Precomputing typology embeddings...")
        
        with torch.no_grad():
            for lang_id in self.languages:
                # Get raw features
                features = self.feature_loader.get_feature_vector(lang_id)
                self.typology_features[lang_id] = features
                
                # Get learned embedding
                embedding = self.model.typology_module.get_embedding(lang_id)
                self.typology_embeddings[lang_id] = embedding.cpu().numpy()
        
        print(f"  Computed embeddings for {len(self.languages)} languages")
    
    def compute_similarity_matrix(
        self,
        languages: Optional[List[str]] = None,
        similarity_type: str = 'embedding',
        metric: str = 'cosine'
    ) -> pd.DataFrame:
        """
        Compute pairwise similarity matrix between languages.
        
        Args:
            languages: Languages to include (None = all)
            similarity_type: 'embedding' or 'features'
            metric: 'cosine', 'euclidean', or 'correlation'
        
        Returns:
            Similarity matrix as DataFrame
        """
        if languages is None:
            languages = self.languages
        
        n = len(languages)
        similarity_matrix = np.zeros((n, n))
        
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if similarity_type == 'embedding':
                    vec1 = self.typology_embeddings[lang1]
                    vec2 = self.typology_embeddings[lang2]
                else:
                    vec1 = self.typology_features[lang1]
                    vec2 = self.typology_features[lang2]
                
                if metric == 'cosine':
                    sim = 1 - cosine(vec1, vec2)
                elif metric == 'euclidean':
                    sim = 1 / (1 + euclidean(vec1, vec2))
                elif metric == 'correlation':
                    sim = pearsonr(vec1, vec2)[0]
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                similarity_matrix[i, j] = sim
        
        return pd.DataFrame(
            similarity_matrix,
            index=languages,
            columns=languages
        )
    
    def analyze_transfer_correlation(
        self,
        performance_data: Dict[str, float],
        source_language: str = 'eng'
    ) -> Dict[str, Any]:
        """
        Analyze correlation between typological similarity and transfer performance.
        
        Args:
            performance_data: {language: accuracy} dictionary
            source_language: Source language for transfer
        
        Returns:
            Correlation analysis results
        """
        languages = list(performance_data.keys())
        
        # Compute similarities to source language
        similarities = []
        performances = []
        
        for lang in languages:
            if lang == source_language:
                continue
            
            # Typological similarity
            emb1 = self.typology_embeddings[source_language]
            emb2 = self.typology_embeddings[lang]
            sim = 1 - cosine(emb1, emb2)
            
            similarities.append(sim)
            performances.append(performance_data[lang])
        
        # Compute correlation
        pearson_r, pearson_p = pearsonr(similarities, performances)
        spearman_r, spearman_p = spearmanr(similarities, performances)
        
        results = {
            'source_language': source_language,
            'pearson_correlation': pearson_r,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_r,
            'spearman_p_value': spearman_p,
            'data': {
                'languages': languages,
                'similarities': similarities,
                'performances': performances
            }
        }
        
        return results
    
    def cluster_languages(
        self,
        languages: Optional[List[str]] = None,
        n_clusters: int = 5,
        method: str = 'kmeans'
    ) -> Dict[str, Any]:
        """
        Cluster languages based on typological features.
        
        Args:
            languages: Languages to cluster
            n_clusters: Number of clusters
            method: 'kmeans' or 'hierarchical'
        
        Returns:
            Clustering results
        """
        if languages is None:
            languages = self.languages
        
        # Prepare embedding matrix
        embeddings = np.array([
            self.typology_embeddings[lang]
            for lang in languages
        ])
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(embeddings)
            
            # Organize results
            clusters = defaultdict(list)
            for lang, label in zip(languages, labels):
                clusters[f'Cluster_{label}'].append(lang)
            
            return {
                'method': 'kmeans',
                'n_clusters': n_clusters,
                'clusters': dict(clusters),
                'labels': labels.tolist(),
                'languages': languages
            }
        
        elif method == 'hierarchical':
            linkage_matrix = linkage(embeddings, method='ward')
            
            return {
                'method': 'hierarchical',
                'linkage_matrix': linkage_matrix,
                'languages': languages
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")


# ============================================================================
# ROUTING PATTERN ANALYZER
# ============================================================================

class RoutingPatternAnalyzer:
    """
    Analyzes adapter routing patterns to understand specialization
    and language-adapter relationships.
    """
    
    def __init__(self, model: CompleteTADRModel, device: str = 'cuda'):
        """
        Args:
            model: Trained TADR model
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        
        # Handle different model structures
        if hasattr(model, 'multi_adapters'):
            self.num_adapters = model.multi_adapters[0].num_adapters
        elif hasattr(model, 'tadr_layers'):
            # CompleteTADRModel structure
            self.num_adapters = model.tadr_layers[0].multi_adapter.num_adapters
        else:
            # Fallback
            self.num_adapters = 8  # Default value
            print(f"Warning: Could not determine num_adapters from model, using default: {self.num_adapters}")
                
        # Storage for routing patterns
        self.language_routing = defaultdict(list)
        self.adapter_specialization = defaultdict(lambda: defaultdict(int))
    
    def collect_routing_patterns(
        self,
        dataloader,
        max_samples: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Collect routing patterns from a dataset.
        
        Args:
            dataloader: DataLoader with samples
            max_samples: Maximum samples to process
        
        Returns:
            Average routing weights per language
        """
        self.model.eval()
        
        samples_processed = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_processed >= max_samples:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                language_ids = batch['language_ids']
                
                outputs = self.model(
                    lang_ids=language_ids,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_routing_info=True
                )
                
                if 'routing_info' in outputs:
                    # Handle routing info structure
                    routing_info = outputs['routing_info']
                    
                    # routing_info is a list of dicts (one per layer), not per sample
                    if routing_info and len(routing_info) > 0:
                        # Get batch size from first layer's weights
                        batch_size = routing_info[0]['weights'].size(0)
                        
                        for batch_idx, lang_id in enumerate(language_ids):
                            # Collect weights for this sample across all layers
                            layer_weights = []
                            for layer_info in routing_info:
                                if 'weights' in layer_info:
                                    # Extract weights for this sample in the batch
                                    weights = layer_info['weights'][batch_idx]
                                    layer_weights.append(weights)
                            
                            if layer_weights:
                                # Average across layers
                                avg_weights = torch.stack(layer_weights).mean(dim=0)
                                self.language_routing[lang_id].append(
                                    avg_weights.cpu().numpy()
                                )
                        
                        self.language_routing[lang_id].append(
                            avg_weights.cpu().numpy()
                        )
                
                samples_processed += len(language_ids)
        
        # Compute averages
        avg_routing = {}
        for lang_id, weights_list in self.language_routing.items():
            avg_routing[lang_id] = np.mean(weights_list, axis=0)
        
        return avg_routing
    
    def analyze_adapter_specialization(
        self,
        routing_patterns: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Analyze what each adapter specializes in.
        
        Args:
            routing_patterns: Average routing weights per language
        
        Returns:
            DataFrame with adapter specialization analysis
        """
        adapter_data = []
        
        for adapter_idx in range(self.num_adapters):
            # Languages that use this adapter most
            language_weights = {
                lang: weights[adapter_idx]
                for lang, weights in routing_patterns.items()
            }
            
            # Sort by weight
            top_languages = sorted(
                language_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Compute statistics
            weights = list(language_weights.values())
            
            adapter_data.append({
                'adapter_id': f'Adapter_{adapter_idx}',
                'mean_weight': np.mean(weights),
                'std_weight': np.std(weights),
                'max_weight': np.max(weights),
                'top_languages': [lang for lang, _ in top_languages],
                'top_weights': [weight for _, weight in top_languages],
                'usage_entropy': self._compute_entropy(weights)
            })
        
        return pd.DataFrame(adapter_data)
    
    def _compute_entropy(self, weights: List[float]) -> float:
        """Compute entropy of weight distribution."""
        weights = np.array(weights)
        weights = weights / weights.sum()
        epsilon = 1e-10
        return -np.sum(weights * np.log(weights + epsilon))
    
    def compute_adapter_similarity(
        self,
        routing_patterns: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute similarity between adapters based on usage patterns.
        
        Args:
            routing_patterns: Routing patterns per language
        
        Returns:
            Adapter similarity matrix
        """
        # Create adapter usage matrix (adapters x languages)
        languages = list(routing_patterns.keys())
        adapter_matrix = np.zeros((self.num_adapters, len(languages)))
        
        for lang_idx, lang in enumerate(languages):
            adapter_matrix[:, lang_idx] = routing_patterns[lang]
        
        # Compute cosine similarity between adapters
        similarity_matrix = np.zeros((self.num_adapters, self.num_adapters))
        
        for i in range(self.num_adapters):
            for j in range(self.num_adapters):
                sim = 1 - cosine(adapter_matrix[i], adapter_matrix[j])
                similarity_matrix[i, j] = sim
        
        return similarity_matrix


# ============================================================================
# VISUALIZATION TOOLS
# ============================================================================

class TADRVisualizer:
    """Advanced visualization tools for TADR analysis."""
    
    @staticmethod
    def plot_typology_space(
        embeddings: Dict[str, np.ndarray],
        performance: Optional[Dict[str, float]] = None,
        method: str = 'tsne',
        save_path: Optional[str] = None
    ):
        """
        Visualize languages in typological embedding space.
        
        Args:
            embeddings: Language embeddings
            performance: Optional performance scores for coloring
            method: 'tsne' or 'pca'
            save_path: Path to save figure
        """
        languages = list(embeddings.keys())
        X = np.array([embeddings[lang] for lang in languages])
        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
        else:
            reducer = PCA(n_components=2)
            X_reduced = reducer.fit_transform(X)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if performance:
            colors = [performance.get(lang, 0) for lang in languages]
            scatter = ax.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                c=colors,
                cmap='viridis',
                s=100,
                alpha=0.7
            )
            plt.colorbar(scatter, label='Performance')
        else:
            ax.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                s=100,
                alpha=0.7
            )
        
        # Add labels
        for i, lang in enumerate(languages):
            ax.annotate(
                lang,
                (X_reduced[i, 0], X_reduced[i, 1]),
                fontsize=8,
                alpha=0.8
            )
        
        ax.set_title(f'Typological Embedding Space ({method.upper()})', fontsize=14)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_language_similarity_network(
        similarity_matrix: pd.DataFrame,
        threshold: float = 0.7,
        save_path: Optional[str] = None
    ):
        """
        Plot language similarity as a network graph.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            threshold: Minimum similarity for edge creation
            save_path: Path to save figure
        """
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        languages = list(similarity_matrix.index)
        G.add_nodes_from(languages)
        
        # Add edges for high similarity
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i < j:
                    sim = similarity_matrix.iloc[i, j]
                    if sim > threshold:
                        G.add_edge(lang1, lang2, weight=sim)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=1000,
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(
            G, pos,
            width=[w * 3 for w in weights],
            alpha=0.5,
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title(f'Language Similarity Network (threshold={threshold})', fontsize=14)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_adapter_specialization(
        specialization_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Visualize adapter specialization patterns.
        
        Args:
            specialization_df: DataFrame from analyze_adapter_specialization
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Mean weight distribution
        axes[0, 0].bar(
            specialization_df['adapter_id'],
            specialization_df['mean_weight'],
            color='steelblue'
        )
        axes[0, 0].set_title('Average Adapter Usage', fontsize=12)
        axes[0, 0].set_xlabel('Adapter')
        axes[0, 0].set_ylabel('Mean Weight')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Usage entropy
        axes[0, 1].bar(
            specialization_df['adapter_id'],
            specialization_df['usage_entropy'],
            color='coral'
        )
        axes[0, 1].set_title('Adapter Usage Diversity (Entropy)', fontsize=12)
        axes[0, 1].set_xlabel('Adapter')
        axes[0, 1].set_ylabel('Entropy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Top languages per adapter
        adapter_lang_matrix = np.zeros((
            len(specialization_df),
            max(len(langs) for langs in specialization_df['top_languages'])
        ))
        
        for i, row in specialization_df.iterrows():
            for j, weight in enumerate(row['top_weights'][:3]):
                adapter_lang_matrix[i, j] = weight
        
        im = axes[1, 0].imshow(adapter_lang_matrix, cmap='YlOrRd', aspect='auto')
        axes[1, 0].set_title('Top Language Weights per Adapter', fontsize=12)
        axes[1, 0].set_xlabel('Top Language Rank')
        axes[1, 0].set_ylabel('Adapter')
        axes[1, 0].set_yticks(range(len(specialization_df)))
        axes[1, 0].set_yticklabels(specialization_df['adapter_id'])
        plt.colorbar(im, ax=axes[1, 0])
        
        # Variance in weights
        axes[1, 1].scatter(
            specialization_df['mean_weight'],
            specialization_df['std_weight'],
            s=100,
            alpha=0.6
        )
        axes[1, 1].set_title('Mean vs Variance in Adapter Usage', fontsize=12)
        axes[1, 1].set_xlabel('Mean Weight')
        axes[1, 1].set_ylabel('Std Weight')
        
        for i, txt in enumerate(specialization_df['adapter_id']):
            axes[1, 1].annotate(
                txt.replace('Adapter_', 'A'),
                (specialization_df['mean_weight'].iloc[i],
                 specialization_df['std_weight'].iloc[i]),
                fontsize=8
            )
        
        plt.suptitle('Adapter Specialization Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_transfer_correlation(
        correlation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """
        Plot correlation between typological similarity and transfer performance.
        
        Args:
            correlation_results: Results from analyze_transfer_correlation
            save_path: Path to save figure
        """
        data = correlation_results['data']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(
            data['similarities'],
            data['performances'],
            s=100,
            alpha=0.6,
            c=range(len(data['similarities'])),
            cmap='viridis'
        )
        
        # Add trend line
        z = np.polyfit(data['similarities'], data['performances'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(data['similarities']), max(data['similarities']), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, label='Trend')
        
        # Add correlation text
        ax.text(
            0.05, 0.95,
            f"Pearson r = {correlation_results['pearson_correlation']:.3f}\n"
            f"Spearman ρ = {correlation_results['spearman_correlation']:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        ax.set_xlabel('Typological Similarity', fontsize=12)
        ax.set_ylabel('Transfer Performance', fontsize=12)
        ax.set_title(
            f'Typological Similarity vs Transfer Performance\n'
            f'(Source: {correlation_results["source_language"]})',
            fontsize=14
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.show()
    
    @staticmethod
    def create_interactive_routing_plot(
        routing_patterns: Dict[str, np.ndarray],
        output_path: str = 'routing_interactive.html'
    ):
        """
        Create interactive routing visualization with Plotly.
        
        Args:
            routing_patterns: Routing patterns per language
            output_path: Path to save HTML
        """
        languages = list(routing_patterns.keys())
        n_adapters = len(next(iter(routing_patterns.values())))
        
        # Prepare data
        data = []
        for lang in languages:
            weights = routing_patterns[lang]
            data.append(
                go.Bar(
                    name=lang,
                    x=[f'A{i}' for i in range(n_adapters)],
                    y=weights
                )
            )
        
        # Create figure
        fig = go.Figure(data=data)
        
        fig.update_layout(
            title='Interactive Adapter Routing Patterns',
            xaxis_title='Adapter',
            yaxis_title='Routing Weight',
            barmode='group',
            height=600,
            hovermode='x unified'
        )
        
        fig.write_html(output_path)
        print(f"  Interactive plot saved: {output_path}")


# ============================================================================
# COMPREHENSIVE ANALYSIS PIPELINE
# ============================================================================

class TADRAnalysisPipeline:
    """
    Complete analysis pipeline combining all analysis components.
    """
    
    def __init__(
        self,
        model: CompleteTADRModel,
        feature_file: str = 'wals_features.csv',
        device: str = 'cuda'
    ):
        """
        Args:
            model: Trained TADR model
            feature_file: Path to WALS features
            device: Device to use
        """
        self.model = model
        self.device = device
        
        # Initialize analyzers
        self.typology_analyzer = TypologicalSimilarityAnalyzer(
            model, feature_file, device
        )
        self.routing_analyzer = RoutingPatternAnalyzer(model, device)
        self.visualizer = TADRVisualizer()
        
        # Results storage
        self.results = {}
    
    def run_complete_analysis(
        self,
        test_dataloader,
        performance_data: Dict[str, float],
        output_dir: str = './analysis_results',
        source_language: str = 'eng'
    ) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Args:
            test_dataloader: DataLoader for test data
            performance_data: Language performance scores
            output_dir: Output directory
            source_language: Source language for transfer analysis
        
        Returns:
            Complete analysis results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("RUNNING COMPREHENSIVE TADR ANALYSIS")
        print("="*70)
        
        # ========================================================================
        # 1. TYPOLOGICAL SIMILARITY ANALYSIS
        # ========================================================================
        
        print("\n📊 Step 1: Typological Similarity Analysis")
        print("-"*40)
        
        # Compute similarity matrix
        languages = list(performance_data.keys())
        sim_matrix = self.typology_analyzer.compute_similarity_matrix(
            languages=languages,
            similarity_type='embedding',
            metric='cosine'
        )
        
        # Save similarity matrix
        sim_matrix.to_csv(
            os.path.join(output_dir, 'similarity_matrix.csv')
        )
        print(f"  Similarity matrix computed for {len(languages)} languages")
        
        # Visualize typology space
        self.visualizer.plot_typology_space(
            self.typology_analyzer.typology_embeddings,
            performance=performance_data,
            method='tsne',
            save_path=os.path.join(output_dir, 'typology_space.png')
        )
        
        # ========================================================================
        # 2. TRANSFER CORRELATION ANALYSIS
        # ========================================================================
        
        print("\n🔄 Step 2: Transfer Correlation Analysis")
        print("-"*40)
        
        correlation_results = self.typology_analyzer.analyze_transfer_correlation(
            performance_data=performance_data,
            source_language=source_language
        )
        
        print(f"  Source language: {source_language}")
        print(f"  Pearson correlation: {correlation_results['pearson_correlation']:.3f}")
        print(f"  Spearman correlation: {correlation_results['spearman_correlation']:.3f}")
        
        # Visualize correlation
        self.visualizer.plot_transfer_correlation(
            correlation_results,
            save_path=os.path.join(output_dir, 'transfer_correlation.png')
        )
        
        # ========================================================================
        # 3. LANGUAGE CLUSTERING
        # ========================================================================
        
        print("\n🗂️ Step 3: Language Clustering")
        print("-"*40)
        
        clustering_results = self.typology_analyzer.cluster_languages(
            languages=languages,
            n_clusters=min(5, len(languages)),
            method='kmeans'
        )
        
        print(f"  Clustered {len(languages)} languages into {clustering_results['n_clusters']} groups")
        for cluster_name, cluster_langs in clustering_results['clusters'].items():
            print(f"    {cluster_name}: {cluster_langs}")
        
        # ========================================================================
        # 4. ROUTING PATTERN ANALYSIS
        # ========================================================================
        
        print("\n🔀 Step 4: Routing Pattern Analysis")
        print("-"*40)
        
        # Collect routing patterns
        routing_patterns = self.routing_analyzer.collect_routing_patterns(
            test_dataloader,
            max_samples=1000
        )
        
        print(f"  Collected routing patterns for {len(routing_patterns)} languages")
        
        # Analyze adapter specialization
        specialization_df = self.routing_analyzer.analyze_adapter_specialization(
            routing_patterns
        )
        specialization_df.to_csv(
            os.path.join(output_dir, 'adapter_specialization.csv'),
            index=False
        )
        
        print("\n  Adapter Specialization:")
        for _, row in specialization_df.iterrows():
            print(f"    {row['adapter_id']}: "
                  f"mean_weight={row['mean_weight']:.3f}, "
                  f"top_langs={row['top_languages'][:3]}")
        
        # Visualize specialization
        self.visualizer.plot_adapter_specialization(
            specialization_df,
            save_path=os.path.join(output_dir, 'adapter_specialization.png')
        )
        
        # ========================================================================
        # 5. NETWORK VISUALIZATION
        # ========================================================================
        
        print("\n🌐 Step 5: Network Visualization")
        print("-"*40)
        
        # Language similarity network
        self.visualizer.plot_language_similarity_network(
            sim_matrix,
            threshold=0.7,
            save_path=os.path.join(output_dir, 'language_network.png')
        )
        
        # Interactive routing visualization
        self.visualizer.create_interactive_routing_plot(
            routing_patterns,
            output_path=os.path.join(output_dir, 'routing_interactive.html')
        )
        
        # ========================================================================
        # 6. COMPILE RESULTS
        # ========================================================================
        
        print("\n📝 Step 6: Compiling Results")
        print("-"*40)
        
        self.results = {
            'similarity_matrix': sim_matrix.to_dict(),
            'correlation_analysis': correlation_results,
            'clustering': clustering_results,
            'routing_patterns': {k: v.tolist() for k, v in routing_patterns.items()},
            'adapter_specialization': specialization_df.to_dict('records'),
            'performance_data': performance_data
        }

        def convert_to_json_serializable(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        # Save complete results
        results_path = os.path.join(output_dir, 'complete_analysis.json')
        serializable_results = convert_to_json_serializable(self.results)
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"  Complete results saved to: {results_path}")
        
        # Generate report
        self._generate_report(output_dir)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {output_dir}")
        
        return self.results
    
    def _generate_report(self, output_dir: str):
        """Generate comprehensive analysis report."""
        report_path = os.path.join(output_dir, 'analysis_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# TADR Analysis Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            f.write("## 1. Executive Summary\n\n")
            
            # Key findings
            corr = self.results['correlation_analysis']
            f.write(f"- **Typology-Transfer Correlation**: "
                   f"r={corr['pearson_correlation']:.3f} "
                   f"(p={corr['pearson_p_value']:.4f})\n")
            
            f.write(f"- **Language Clusters**: "
                   f"{len(self.results['clustering']['clusters'])} identified\n")
            
            # Adapter usage
            spec_df = pd.DataFrame(self.results['adapter_specialization'])
            most_used = spec_df.loc[spec_df['mean_weight'].idxmax()]
            f.write(f"- **Most Used Adapter**: {most_used['adapter_id']} "
                   f"(mean weight={most_used['mean_weight']:.3f})\n\n")
            
            f.write("## 2. Typological Analysis\n\n")
            f.write("### Language Clusters\n\n")
            for cluster, langs in self.results['clustering']['clusters'].items():
                f.write(f"- **{cluster}**: {', '.join(langs)}\n")
            
            f.write("\n### Transfer Correlation\n\n")
            f.write("Typological similarity shows ")
            if abs(corr['pearson_correlation']) > 0.5:
                f.write("**strong** ")
            elif abs(corr['pearson_correlation']) > 0.3:
                f.write("**moderate** ")
            else:
                f.write("**weak** ")
            f.write(f"correlation with transfer performance.\n\n")
            
            f.write("## 3. Adapter Specialization\n\n")
            f.write("### Top Languages per Adapter\n\n")
            for adapter in self.results['adapter_specialization']:
                f.write(f"**{adapter['adapter_id']}**: ")
                f.write(f"{', '.join(adapter['top_languages'][:3])}\n")
            
            f.write("\n## 4. Visualizations\n\n")
            f.write("The following visualizations have been generated:\n\n")
            f.write("- `typology_space.png`: Language embeddings visualization\n")
            f.write("- `transfer_correlation.png`: Similarity-performance correlation\n")
            f.write("- `adapter_specialization.png`: Adapter usage patterns\n")
            f.write("- `language_network.png`: Language similarity network\n")
            f.write("- `routing_interactive.html`: Interactive routing visualization\n")
        
        print(f"  Report generated: {report_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_analysis(
    checkpoint_path: str,
    test_results_path: str,
    data_dir: str = './data_zeroshot',
    output_dir: str = './analysis_results',
    model_name: str = 'xlm-roberta-base',
    batch_size: int = 32,
    device: str = None
):
    """
    Run complete TADR analysis pipeline.
    
    Args:
        checkpoint_path: Path to trained model
        test_results_path: Path to test results from Step 8
        data_dir: Data directory
        output_dir: Output directory for analysis
        model_name: Base model name
        batch_size: Batch size
        device: Device to use
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "🔬"*35)
    print("TADR ANALYSIS & INTERPRETABILITY")
    print("🔬"*35 + "\n")
    
    # Load model
    print("Loading model...")
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
    
    # Load test results
    print("Loading test results...")
    with open(test_results_path, 'rb') as f:
        test_results = pickle.load(f)
    
    # Extract performance data
    performance_data = {}
    if 'per_language' in test_results:
        for lang, metrics in test_results['per_language'].items():
            performance_data[lang] = metrics['accuracy']
    # Otherwise use seen_languages and unseen_languages (new structure)
    else:
        # Add seen languages
        if 'seen_languages' in test_results:
            for lang, metrics in test_results['seen_languages'].items():
                performance_data[lang] = metrics['accuracy']
        
        # Add unseen languages
        if 'unseen_languages' in test_results:
            for lang, metrics in test_results['unseen_languages'].items():
                performance_data[lang] = metrics['accuracy']
    
    print(f"Extracted performance data for {len(performance_data)} languages")

    
    # Load test data
    from step7_training_loop import load_preprocessed_data, MultilingualDataset, collate_fn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    
    _, _, test_data = load_preprocessed_data(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    test_dataset = MultilingualDataset(
        texts=test_data['texts'],
        labels=test_data['labels'],
        language_ids=test_data['lang_ids'],
        tokenizer=tokenizer,
        max_length=128
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Run analysis
    pipeline = TADRAnalysisPipeline(model, device=device)
    results = pipeline.run_complete_analysis(
        test_dataloader=test_loader,
        performance_data=performance_data,
        output_dir=output_dir,
        source_language='eng'
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TADR Analysis & Interpretability'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./tadr_checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--test_results',
        type=str,
        default='./zero_shot_results/tadr_zero_shot_complete.pkl',
        help='Path to test results from Step 8'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./analysis_results',
        help='Output directory'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    results = run_analysis(
        checkpoint_path=args.checkpoint,
        test_results_path=args.test_results,
        output_dir=args.output_dir,
        device=args.device
    )