"""
TADR Framework - Step 5: Integration Layer
Complete TADR model combining all components with advanced dynamic routing
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
import warnings

# Import all previous components
from step1_typology_module import TypologicalFeatureLoader, TypologyFeatureModule
from step2_base_model import ModelWithAdapterSlots, DEFAULT_MODEL_NAME
from step3_adapter_modules import MultiAdapterModule
from step4_routing_network import DynamicRouter, ContextExtractor, LoadBalancingLoss


# ============================================================================
# TADR LAYER
# ============================================================================

class TADRLayer(nn.Module):
    """
    Single TADR layer combining transformer layer with dynamic adapters and routing.
    
    Flow:
        Input → Transformer Layer → Extract Context → Route → Apply Adapters → Output
    """
    
    def __init__(
        self,
        transformer_layer: nn.Module,
        multi_adapter: MultiAdapterModule,
        router: DynamicRouter,
        context_extractor: ContextExtractor,
        apply_after_attention: bool = False,
        apply_after_ffn: bool = True
    ):
        """
        Args:
            transformer_layer: Original transformer layer from base model
            multi_adapter: MultiAdapterModule with K adapters
            router: DynamicRouter for computing routing weights
            context_extractor: Extracts context features from hidden states
            apply_after_attention: Whether to apply adapters after attention
            apply_after_ffn: Whether to apply adapters after FFN (standard)
        """
        super().__init__()
        
        self.transformer_layer = transformer_layer
        self.multi_adapter = multi_adapter
        self.router = router
        self.context_extractor = context_extractor
        
        self.apply_after_attention = apply_after_attention
        self.apply_after_ffn = apply_after_ffn
        
        # Cache for analysis
        self.last_routing_weights = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        typology_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        extended_attention_mask: Optional[torch.Tensor] = None,
        return_routing_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through TADR layer.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            typology_embedding: (batch_size, typology_dim)
            attention_mask: (batch_size, seq_len)
            return_routing_info: Whether to return routing diagnostics
        
        Returns:
            output_hidden_states: (batch_size, seq_len, hidden_size)
            routing_info: Dict with routing weights and statistics (if requested)
        """
        # Extract context features for routing
        context_features = self.context_extractor(hidden_states, attention_mask)
        
        # Compute routing weights from typology + context
        routing_weights, logits = self.router(
            typology_embedding, 
            context_features,
            return_logits=True
        )
        self.last_routing_weights = routing_weights.detach()
        
        # Pass through original transformer layer
        layer_output = self.transformer_layer(
            hidden_states, 
            attention_mask = extended_attention_mask
        )[0]
        
        # Apply routed adapters (standard placement: after FFN)
        if self.apply_after_ffn:
            layer_output = self.multi_adapter(layer_output, routing_weights)
        
        # Prepare routing info if requested
        routing_info = None
        if return_routing_info:
            epsilon = 1e-10
            routing_info = {
                'weights': routing_weights,
                'logits': logits,
                'entropy': -(routing_weights * torch.log(routing_weights + epsilon)).sum(dim=-1),
                'sparsity': (routing_weights > 1e-5).sum(dim=-1).float(),
                'top_adapter': routing_weights.argmax(dim=-1),
                'max_weight': routing_weights.max(dim=-1)[0]
            }
        
        return layer_output, routing_info


# ============================================================================
# COMPLETE TADR MODEL
# ============================================================================

class CompleteTADRModel(nn.Module):
    """
    Complete TADR Model: Typology-Aware Dynamic Routing for Multilingual NLP.
    
    This is the advanced version with context-aware routing (Step 5).
    For simpler typology-only routing, use TADRModel from step3_adapters.py.
    
    Architecture:
        Input → Embeddings → [TADR Layer 1] → ... → [TADR Layer N] → Output
        
    Each TADR layer:
        1. Passes through transformer layer
        2. Extracts context from hidden states
        3. Routes based on typology + context
        4. Applies weighted adapters
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        feature_file: str = 'wals_features.csv',
        # Typology config
        typology_embedding_dim: int = 128,
        typology_hidden_dim: int = 256,
        # Adapter config
        num_adapters: int = 10,
        adapter_bottleneck_size: int = 64,
        adapter_non_linearity: str = "relu",
        adapter_dropout: float = 0.1,
        # Router config
        router_hidden_dims: List[int] = None,
        router_dropout: float = 0.1,
        gating_type: str = "softmax",
        gating_config: Optional[Dict] = None,
        pooling_type: str = "cls",
        # General config
        num_adapter_layers: Optional[int] = None,
        num_classes: Optional[int] = None,
        device: Optional[str] = None,
        unfreezing_strategy: str = "full",
    ):
        super().__init__()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        router_hidden_dims = router_hidden_dims or [256, 128]
        
        print("="*70)
        print("Initializing Complete TADR Model (Advanced Version)")
        print("="*70)
        
        # ========================
        # Step 1: Typology Module
        # ========================
        feature_loader = TypologicalFeatureLoader(
            feature_file=feature_file,
            missing_value_strategy='mean'
        )
        self.typology_module = TypologyFeatureModule(
            feature_loader=feature_loader,
            embedding_dim=typology_embedding_dim,
            hidden_dim=typology_hidden_dim
        )
        
        # ========================
        # Step 2: Base Model
        # ========================
        self.base_model_wrapper = ModelWithAdapterSlots(
            model_name=model_name,
            freeze_base=True,
            num_adapter_layers=num_adapter_layers,
            device=self.device
        )
        
        self.base_model = self.base_model_wrapper.model
        self.config = self.base_model_wrapper.config
        self.hidden_size = self.base_model_wrapper.hidden_size
        self.num_layers = self.base_model_wrapper.num_layers
        self.adapter_layer_indices = self.base_model_wrapper.adapter_layer_indices
        self.num_adapter_layers = len(self.adapter_layer_indices)
        
        # ========================
        # Steps 3 & 4: TADR Layers
        # ========================
        self.tadr_layers = nn.ModuleList()
        
        for layer_idx in self.adapter_layer_indices:
            # Multi-adapter module
            multi_adapter = MultiAdapterModule(
                num_adapters=num_adapters,
                hidden_size=self.hidden_size,
                bottleneck_size=adapter_bottleneck_size,
                non_linearity=adapter_non_linearity,
                dropout=adapter_dropout
            )
            
            # Dynamic router (typology + context)
            router = DynamicRouter(
                typology_dim=typology_embedding_dim,
                context_dim=self.hidden_size,
                num_adapters=num_adapters,
                hidden_dims=router_hidden_dims,
                dropout=router_dropout,
                gating_type=gating_type,
                gating_config=gating_config or {}
            )
            
            # Context extractor
            context_extractor = ContextExtractor(
                hidden_size=self.hidden_size,
                pooling_type=pooling_type
            )
            
            # Get transformer layer
            transformer_layer = self.base_model_wrapper.encoder_layers[layer_idx]
            
            # Create TADR layer
            tadr_layer = TADRLayer(
                transformer_layer=transformer_layer,
                multi_adapter=multi_adapter,
                router=router,
                context_extractor=context_extractor,
                apply_after_ffn=True
            )
            
            self.tadr_layers.append(tadr_layer)
        
        # ========================
        # Classification head
        # ========================
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )
            for module in self.classifier.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    nn.init.zeros_(module.bias)
        else:
            self.classifier = None

        # ========================
        # Apply unfreezing strategy
        # ========================
        if unfreezing_strategy != 'minimal':
            self.base_model_wrapper.get_unfreezing_strategy(unfreezing_strategy)
        
        # Move to device
        self.to(self.device)
        
        # Print summary
        self._print_summary()
    
    def apply_unfreezing_strategy(self, strategy: str):
        self.base_model_wrapper.get_unfreezing_strategy(strategy)
        # Print updated parameter count
        trainable = self.count_trainable_parameters()
        total = self.count_parameters()
        print(f"\n📊 Updated trainable parameters:")
        print(f"  Total: {total:,}")
        print(f"  Trainable: {trainable:,} ({trainable/total*100:.2f}%)")

    
    def _print_summary(self):
        """Print model configuration summary."""
        total_params = self.count_parameters()
        trainable_params = self.count_trainable_parameters()
        adapter_params = self.count_adapter_parameters()
        router_params = self.count_router_parameters()
        
        print(f"\n✅ Model Configuration:")
        print(f"  Base model: {self.base_model_wrapper.model_name}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Total transformer layers: {self.num_layers}")
        print(f"  TADR layers: {self.num_adapter_layers} at indices {self.adapter_layer_indices}")
        
        if self.classifier is not None:
            print(f"  Task: Classification ({self.num_classes} classes)")
        
        print(f"\n📊 Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  ├─ Adapters: {adapter_params:,} ({adapter_params/trainable_params*100:.1f}% of trainable)")
        print(f"  ├─ Routers: {router_params:,} ({router_params/trainable_params*100:.1f}% of trainable)")
        if self.classifier is not None:
            classifier_params = sum(p.numel() for p in self.classifier.parameters())
            print(f"  └─ Classifier: {classifier_params:,}")
        
        print("="*70 + "\n")
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_adapter_parameters(self) -> int:
        """Count adapter parameters."""
        return sum(
            p.numel() 
            for layer in self.tadr_layers 
            for p in layer.multi_adapter.parameters()
        )
    
    def count_router_parameters(self) -> int:
        """Count router parameters."""
        return sum(
            p.numel() 
            for layer in self.tadr_layers 
            for p in layer.router.parameters()
        )
    
    def forward(
        self,
        lang_ids: List[str],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_routing_info: bool = False,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass through complete TADR model.
        
        Args:
            lang_ids: List of language ISO codes for the batch
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            token_type_ids: Token type IDs (batch_size, seq_len)
            return_routing_info: Whether to return routing diagnostics
            return_dict: Whether to return as dict
        
        Returns:
            Dictionary containing:
                - last_hidden_state: Final hidden states
                - pooler_output: Pooled output (CLS token)
                - logits: Classification logits (if classifier exists)
                - hidden_states: All layer hidden states
                - routing_info: Routing information per layer (if requested)
        """
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        
        batch_size = input_ids.size(0)
        
        # Get typology embeddings
        typology_embeddings = self.typology_module(lang_ids)
        typology_embeddings = typology_embeddings.to(self.device)
        
        # Ensure batch size matches
        if typology_embeddings.size(0) == 1 and batch_size > 1:
            typology_embeddings = typology_embeddings.expand(batch_size, -1)
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        extended_attention_mask = self.base_model.get_extended_attention_mask(
            attention_mask, input_ids.size(), device=self.device
        )
        
        # Get embeddings
        hidden_states = self.base_model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        
        all_hidden_states = [hidden_states]
        routing_info_list = []
        
        # Pass through transformer layers with TADR
        encoder_layers = self.base_model_wrapper.encoder_layers
        tadr_idx = 0
        
        for layer_idx, layer_module in enumerate(encoder_layers):
            if layer_idx in self.adapter_layer_indices:
                # Use TADR layer
                hidden_states, routing_info = self.tadr_layers[tadr_idx](
                    hidden_states = hidden_states,
                    typology_embedding = typology_embeddings,
                    attention_mask = attention_mask,
                    extended_attention_mask = extended_attention_mask,
                    return_routing_info = return_routing_info
                )
                
                if return_routing_info:
                    routing_info['layer_idx'] = layer_idx
                    routing_info_list.append(routing_info)
                
                tadr_idx += 1
            else:
                # Use standard transformer layer
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask
                )
                hidden_states = layer_outputs[0]
            
            all_hidden_states.append(hidden_states)
        
        # Pooler
        pooler_output = None
        if hasattr(self.base_model, 'pooler') and self.base_model.pooler is not None:
            pooler_output = self.base_model.pooler(hidden_states)
        else:
            pooler_output = hidden_states[:, 0]
        
        # Classification
        logits = None
        if self.classifier is not None:
            logits = self.classifier(pooler_output)
        
        # Prepare outputs
        outputs = {
            'last_hidden_state': hidden_states,
            'pooler_output': pooler_output,
            'hidden_states': tuple(all_hidden_states)
        }
        
        if logits is not None:
            outputs['logits'] = logits
        
        if return_routing_info:
            outputs['routing_info'] = routing_info_list
        
        return outputs
    
    def analyze_routing(
        self,
        lang_ids: List[str],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Analyze routing patterns for given input.
        
        Args:
            lang_ids: List of language codes
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Dictionary with routing analysis across all layers
        """
        with torch.no_grad():
            outputs = self.forward(
                lang_ids=lang_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_routing_info=True
            )
        
        routing_info = outputs['routing_info']
        
        # Aggregate statistics
        analysis = {
            'per_layer': routing_info,
            'avg_entropy': torch.stack([info['entropy'] for info in routing_info]).mean().item(),
            'avg_sparsity': torch.stack([info['sparsity'] for info in routing_info]).mean().item(),
            'adapter_usage': {}
        }
        
        # Count adapter usage across all layers
        adapter_counts = {}
        for info in routing_info:
            for adapter_idx in info['top_adapter'].cpu().tolist():
                adapter_counts[adapter_idx] = adapter_counts.get(adapter_idx, 0) + 1
        
        analysis['adapter_usage'] = adapter_counts
        
        return analysis
    
    def get_routing_weights(
        self,
        lang_ids: List[str],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Get routing weights for all TADR layers.
        
        Returns:
            List of routing weight tensors, one per TADR layer
        """
        with torch.no_grad():
            outputs = self.forward(
                lang_ids=lang_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_routing_info=True
            )
        
        return [info['weights'] for info in outputs['routing_info']]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_tadr_model(
    model_name: str = DEFAULT_MODEL_NAME,
    feature_file: str = 'wals_features.csv',
    num_adapters: int = 8,
    adapter_bottleneck: int = 38,
    num_classes: Optional[int] = None,
    gating_type: str = "softmax",
    num_adapter_layers: int = 4,
    device: Optional[str] = None,
    unfreezing_strategy: str = "full",
) -> CompleteTADRModel:

    gating_config = {}
    if gating_type == "topk":
        gating_config = {"k": 3, "temperature": 1.0}
    elif gating_type == "threshold":
        gating_config = {"threshold": 0.1, "temperature": 1.0}
    
    model = CompleteTADRModel(
        model_name=model_name,
        feature_file=feature_file,
        num_adapters=num_adapters,
        adapter_bottleneck_size=adapter_bottleneck,
        gating_type=gating_type,
        gating_config=gating_config,
        num_adapter_layers=num_adapter_layers,
        num_classes=num_classes,
        device=device
    )
    
    return model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    from wals_preprocessor import prepare_wals_data
    
    print("\n" + "🚀"*35)
    print("TADR Framework - Step 5: Integration Layer")
    print("🚀"*35 + "\n")
    
    # Prepare WALS data if needed
    try:
        prepare_wals_data("cldf-datasets-wals-0f5cd82", "wals_features.csv")
    except:
        print("WALS data already prepared.\n")
    
    # Create complete TADR model
    model = create_tadr_model(
        model_name=DEFAULT_MODEL_NAME,
        feature_file='wals_features.csv',
        num_adapters=8,
        adapter_bottleneck=48,
        num_classes=3,  # Example: 3-way classification
        gating_type="softmax",
        num_adapter_layers=4  # Last 6 layers
    )
    
    print("✅ Complete TADR Model successfully initialized!")
    print("\nModel is ready for training or inference.")
    print("\nKey features:")
    print("  ✓ Context-aware dynamic routing (typology + context)")
    print("  ✓ Parameter-efficient adaptation")
    print("  ✓ Multiple gating strategies")
    print("  ✓ Built-in routing analysis tools")
    
    print("\n" + "="*70)
    print("Complete TADR Framework assembled!")
    print("="*70 + "\n")