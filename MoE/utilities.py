#@title backup utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from math import sqrt

#@title dataset structure
class PhysicsDataset(Dataset):
    def __init__(self, data, patch_size=(16,16,8,4)):
        self.data = data  # [B, X, Y, Z, T]
        self.patch_size = patch_size

    def __getitem__(self, idx):
        # Hierarchical patching for computational efficiency
        patches = self._create_4d_patches(self.data[idx])
        return patches

    def _create_4d_patches(self, data):
        # Efficient 4D patching with overlap
        px, py, pz, pt = self.patch_size
        patches = F.unfold(data, kernel_size=(px,py,pz,pt),
                          stride=(px//2,py//2,pz//2,pt//2))
        return patches

class JointDataset(Dataset):
    def __init__(self, datasets, patch_size=(16,16,8,4)):
        """
        Initialize JointDataset with multiple PhysicsDataset instances

        Args:
            datasets (list): List of PhysicsDataset instances
            patch_size (tuple): Patch dimensions (px,py,pz,pt)
        """
        if len(datasets) != 4:
            raise ValueError(f"Expected 4 physical datasets, got {len(datasets)}")

        # Verify all datasets have same dimensions
        first_data_shape = datasets[0].data.shape
        for i, dataset in enumerate(datasets[1:], 1): # the enumeration starts from 1
            if dataset.data.shape != first_data_shape:
                raise ValueError(f"Dataset {i} shape {dataset.data.shape} "
                              f"doesn't match first dataset shape {first_data_shape}")

            if dataset.patch_size != patch_size:
                raise ValueError(f"Dataset {i} patch size {dataset.patch_size} "
                              f"doesn't match required patch size {patch_size}")

        self.datasets = datasets
        self.patch_size = patch_size

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        # Get patches from each dataset and stack along last dimension
        patches_list = []
        for dataset in self.datasets:
            # Shape: [B, num_patches, px*py*pz*pt]
            patches = dataset[idx]
            # Add new dimension: [B, num_patches, px*py*pz*pt, 1]
            patches = patches.unsqueeze(-1)
            patches_list.append(patches)

        # Stack along last dimension to get [B, num_patches, px*py*pz*pt, 4]
        combined_patches = torch.cat(patches_list, dim=-1)
        return combined_patches

#@title utilities
class OptimizedSeismicTransformer(nn.Module):
    def __init__(self,
                 input_dim=5,
                 d_model=512,
                 nhead=8,
                 num_layers=6,
                 output_dim=2,
                 local_window=7,
                 enable_gradient_checkpointing=False,
                 enable_spectral_norm=False,
                 enable_uncertainty=True,
                 **kwargs):
        super().__init__()

        # Core model configuration
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.local_window = local_window

        # Advanced options
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_spectral_norm = enable_spectral_norm
        self.enable_uncertainty = enable_uncertainty

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all model components with customizable implementations"""
        self.input_norm = self._create_input_normalization()
        self.token_embed = self._create_token_embedding()
        self.pos_encoder = self._create_positional_encoder()
        self.transformer_blocks = self._create_transformer_blocks()
        self.output_heads = self._create_output_heads()

    def _create_input_normalization(self):
        """Customizable input normalization"""
        return nn.LayerNorm(self.input_dim)

    def _create_token_embedding(self):
        """Customizable token embedding"""
        return nn.Sequential(
            PhysicsAwareConv3d(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model*2),
            nn.GELU(),
            nn.Linear(self.d_model*2, self.d_model)
        )

    def _create_positional_encoder(self):
        """Customizable positional encoding"""
        return HashGridPositionalEncoding4D(self.d_model)

    def _create_transformer_blocks(self):
        """Customizable transformer blocks"""
        blocks = nn.ModuleList([
            HierarchicalTransformerBlock(
                d_model=self.d_model,
                nhead=self.nhead,
                local_window=self.local_window*(2**i),
                dim_feedforward=4*self.d_model,
                dropout=0.1
            ) for i in range(self.num_layers)
        ])

        if self.enable_gradient_checkpointing:
            for block in blocks:
                block.use_checkpoint = True

        return blocks

    @abstract
    def _create_output_heads(self):
        """Customizable output heads"""
        return nn.ModuleDict({
            'main': PhysicsInformedOutput(
                self.d_model,
                self.output_dim,
                uncertainty=self.enable_uncertainty
            )
        })

    def _validate_input(self, x, pos):
        """Input validation and shape checking"""
        assert len(x.shape) == 6, f"Expected 6D input (B,T,X,Y,Z,F), got {x.shape}"
        assert len(pos.shape) == 6, f"Expected 6D positions (B,T,X,Y,Z,4), got {pos.shape}"
        assert pos.shape[-1] == 4, f"Position should have 4 coordinates, got {pos.shape[-1]}"

    def register_attention_hooks(self):
        """Register hooks for attention visualization"""
        self.attention_maps = []
        def hook_fn(module, input, output):
            self.attention_maps.append(output.detach())

        for block in self.transformer_blocks:
            block.self_attn.register_forward_hook(hook_fn) # visulizing the output from the pe coor?

    def forward(self, x, pos):
        """Forward pass with modular components"""
        self._validate_input(x, pos)

        # Get dimensions
        B, T, X, Y, Z, _ = x.shape

        # Reshape and normalize input
        x = x.view(B, -1, self.input_dim) # be careful with '-1'
        x = self.input_norm(x)

        # Token embedding
        x = self.token_embed(x)

        # Position encoding
        pos_flat = pos.view(B, -1, 4)
        x = self.pos_encoder(x, pos_flat) #HashGridPositionalEncoding4D.forward

        # Transformer processing
        attn_weights = []
        for i, block in enumerate(self.transformer_blocks):
            if self.enable_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, pos_flat, 2**i
                ) # need massive storage?
            else:
                x = block(x, pos_flat, local_scale=2**i)

        # Output processing
        outputs = {}
        for head_name, head in self.output_heads.items():
            if self.enable_uncertainty:
                mean, var = head(x)
                outputs[head_name] = {
                    'mean': mean.view(B, T, X, Y, Z, -1),
                    'variance': var.view(B, T, X, Y, Z, -1)
                }
            else:
                out = head(x)
                outputs[head_name] = out.view(B, T, X, Y, Z, -1)

        return outputs if len(outputs) > 1 else outputs['main'] # len(mean&variance) > 1

    @property
    def device(self):
        """Helper to get model's device"""
        return next(self.parameters()).device

class PhysicsAwareConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depth_conv = nn.Sequential(
            # Depth-wise
            nn.Conv3d(in_channels, in_channels,
                     kernel_size=(1,3,5), padding='same', groups=in_channels),
            nn.BatchNorm3d(in_channels),
            nn.GELU(),
            # Point-wise
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels)
        )

        # Initialize weights for better gradient flow
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias) # zero bias

    def forward(self, x):
        # More efficient memory layout
        x = x.contiguous()
        return self.depth_conv(x.permute(0,5,1,2,3,4)).permute(0,2,3,4,5,1)

# memory efficient way to pe 4D data. or 5D?
class HashGridPositionalEncoding4D(nn.Module):
    def __init__(self, d_model, num_levels=16, level_dim=2, base_resolution=16):
        '''
        d_model: Output embedding dimension
        num_levels: Number of resolution levels (default 16)
        level_dim: Feature dimension per level (default 2)
        base_resolution: Starting resolution (16^4 entries in first hash table)
        '''
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        self.level_dim = level_dim

        # Hash table for each level
        self.hash_tables = nn.ModuleList([
            nn.Embedding(base_resolution ** 4, level_dim) # Creates lookup tables, each with base_resolution ** 4 entries (for 4D space), level_dim features per entry
            for _ in range(num_levels)
        ])

        # Learnable frequency scaling
        self.freq_bands = nn.Parameter(torch.exp(
            torch.linspace(0., log(128), num_levels))) # Creates learnable frequency scaling factors

    #spatial hashing function
    def hash_fn(self, coords, level):
        # Simple but effective spatial hashing
        primes = torch.tensor([1, 2654435761, 805459861, 3674653429]) # Large prime numbers for better hash distribution
        x = (coords * self.freq_bands[level] * primes) % (2**32)
        return torch.bitwise_xor(x[..., 0], # Combines coordinates using XOR operation
               torch.bitwise_xor(x[..., 1],
               torch.bitwise_xor(x[..., 2], x[..., 3]))) # The result maps 4D coordinates to hash table indices

    def forward(self, x, coords):
        B, N = coords.shape[:2]

        # Normalize coordinates to [0, 1]
        coords = coords.clamp(0, 1)

        # Multi-level encoding
        features = []
        for i, table in enumerate(self.hash_tables): # For each resolution level
            hashed = self.hash_fn(coords, i) # Hash the coordinates
            features.append(table(hashed)) # Look up features in the corresponding hash table

        # Combine features
        encoding = torch.cat(features, dim=-1) # Combine features from all levels
        return x + encoding # Add to input features

class HierarchicalTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, local_window,
                 dim_feedforward, dropout=0.1, use_checkpoint=False):
        super().__init__()

        # Configuration
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint

        # Attention modules
        self.local_attn = FlashAttention(
            d_model, nhead, local_window
        )
        self.global_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feed-Forward Network with intermediate activation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Dynamic gating mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )

    def _local_attention(self, x, pos, local_scale):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.local_attn, x, pos, local_scale
            )
        return self.local_attn(x, pos, local_scale)

    def _global_attention(self, x):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                lambda q, k, v: self.global_attn(q, k, v)[0],
                x, x, x
            )
        return self.global_attn(x, x, x)[0]

    def _compute_dynamic_weights(self, x_local, x_global):
        # Compute attention weights based on input features
        avg_features = (x_local + x_global) / 2
        weights = self.gate_net(avg_features.mean(dim=1, keepdim=True))
        return weights[..., 0:1], weights[..., 1:2]

    def forward(self, x, pos, local_scale=1):
        # Input validation
        if not torch.is_tensor(x) or not torch.is_tensor(pos):
            raise ValueError("Inputs must be tensors")

        # Ensure inputs are on the same device
        if x.device != pos.device:
            pos = pos.to(x.device)

        # Local attention branch with residual
        x_local = self._local_attention(x, pos, local_scale)
        x_local = self.norm1(x + self.dropout(x_local))

        # Global attention branch with residual
        x_global = self._global_attention(x_local)
        x_global = self.norm2(x_local + self.dropout(x_global))

        # Dynamic fusion weights
        alpha, beta = self._compute_dynamic_weights(x_local, x_global)

        # Adaptive fusion with learned weights
        x_fused = alpha * x_global + beta * x_local

        # FFN with residual
        if self.use_checkpoint and self.training:
            x_out = torch.utils.checkpoint.checkpoint(self.ffn, x_fused)
        else:
            x_out = self.ffn(x_fused)

        return self.norm3(x_fused + self.dropout(x_out))

    @property
    def device(self):
        return next(self.parameters()).device

class FlashAttention(nn.Module):
    def __init__(self, d_model, nhead, base_window):
        super().__init__()
        self.window = base_window
        self.d_model = d_model
        self.nhead = nhead

        # Single attention module with flash attention
        self.attention = nn.MultiheadAttention(
            d_model, nhead, batch_first=True,
            dropout=0.1
        )

        # Efficient feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(3*d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # Cache for window indices
        self.register_buffer('window_indices', None)

    def create_window_indices(self, size, window_size):
        '''This function generates indices for creating windows over a 2D grid of size sqrt(size) x sqrt(size).
        size: The total number of elements in the grid.
        window_size: The size of the sliding window used to extract patches from the grid.
        '''
        if self.window_indices is None or self.window_indices.shape[1] != size:
            # Precompute indices for efficient windowing
            idx = torch.arange(size)
            self.window_indices = F.unfold( # extract sliding local blocks (windows) from a larger tensor.
                idx.view(1, 1, int(sqrt(size)), int(sqrt(size))),
                kernel_size=window_size, # a square window of window_size x window_size
                stride=window_size//2 # Using window_size//2 means the windows overlap by half their size, creating a sliding effect.
            ) # [1, window_size**2, num_windows]

    def forward(self, x, pos, scale):
        B, S, D = x.shape
        outputs = []

        for i in range(3): # 3 is hard-coded window scales
            w_size = self.window * (2**i) * scale
            self.create_window_indices(S, w_size)

            # Efficient windowing using cached indices
            # [B, num_windows * window_size**2, D]
            windows = x.index_select(1, self.window_indices.view(-1)) # 第一参数：selecting along the second dimension (dim=1)；
                                          # 第二参数：self.window_indices contains the precomputed indices for windows, which was created by F.unfold.
            # [B*num_windows, window_size**2, D]
            windows = windows.view(-1, w_size**2, D)

            # Flash attention
            with torch.cuda.amp.autocast():
                attn_out = F.scaled_dot_product_attention(
                    windows, windows, windows,
                    dropout_p=0.1 if self.training else 0.0
                )

            outputs.append(attn_out.mean(dim=1)) # [B * num_windows, D]

        return self.fusion(torch.cat(outputs, dim=-1)) # [B * num_windows, D]->[B * num_windows, 3*D]->[B * num_windows, D]

class PhysicsInformedOutput(nn.Module):
    def __init__(self, d_model, output_dim, uncertainty=True):
        super().__init__()
        self.uncertainty = uncertainty

        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            PhysicsConstraintLayer(output_dim),
            nn.GELU(),
            nn.Linear(d_model//2, output_dim)
        )

        if uncertainty:
            self.var_head = nn.Sequential(
                nn.Linear(d_model, d_model//2),
                nn.GELU(),
                nn.Linear(d_model//2, output_dim),
                nn.Softplus()
            )

    def forward(self, x):
        mean = self.mean_head(x)
        if not self.uncertainty:
            return mean, None
        return mean, self.var_head(x) + 1e-6  # Prevent NaN

#normalization?
class PhysicsConstraintLayer(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Initialize with known velocity ranges
        self.register_buffer('vp_bounds', torch.tensor([1500, 6000.0]))
        self.register_buffer('vs_bounds', torch.tensor([500, 4000.0]))

    def forward(self, x):
        # Apply physical constraints
        vp_min, vp_max = self.vp_bounds
        vs_min, vs_max = self.vs_bounds

        x[..., 0] = vp_min + (vp_max - vp_min)*torch.sigmoid(x[..., 0])
        x[..., 1] = vs_min + (vs_max - vs_min)*torch.sigmoid(x[..., 1])
        return x

# training strategy
def train_moe(model, train_loader, optimizer, epochs):
    for epoch in range(epochs):
        for batch in train_loader:
            physics_data, target = batch

            # Forward pass
            pred = model(physics_data)

            # Multi-task loss
            recon_loss = F.mse_loss(pred, target)
            physics_consistency_loss = compute_physics_consistency(pred) # compute_physics_consistency to be added
            temporal_coherence_loss = compute_temporal_coherence(pred) # compute_temporal_coherence to be added

            loss = (recon_loss +
                   0.1 * physics_consistency_loss +
                   0.1 * temporal_coherence_loss)

            # Backward pass with gradient checkpointing
            with torch.cuda.amp.autocast():
                loss.backward()
            optimizer.step()

#@title MoE defines

class BasePhysicsTransformer(OptimizedSeismicTransformer):
    """Base class for all physics transformers, inheriting from OptimizedSeismicTransformer"""
    def __init__(self,
                 physics_type,
                 input_dim=5,
                 d_model=512,
                 nhead=8,
                 num_layers=6,
                 output_dim=2, # mean and variance
                 local_window=7,
                 enable_gradient_checkpointing=False,
                 enable_spectral_norm=False,
                 enable_uncertainty=True,
                 **kwargs):
        self.physics_type = physics_type
        super().__init__(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_dim=output_dim,
            local_window=local_window,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_spectral_norm=enable_spectral_norm,
            enable_uncertainty=enable_uncertainty,
            **kwargs
        )

class SeismicExpert(BasePhysicsTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            physics_type="seismic",
            local_window=8,  # Smaller window for high-res features
            **kwargs
        )

    def _create_token_embedding(self):
        """Specialized seismic embedding with wave propagation awareness"""
        return nn.Sequential(
            PhysicsAwareConv3d(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            # Additional seismic-specific layers
            nn.Conv3d(self.d_model, self.d_model,
                     kernel_size=(3,3,3), padding='same', groups=self.d_model),
            nn.BatchNorm3d(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )

    def _create_output_heads(self):
        """Seismic-specific output with velocity constraints"""
        return nn.ModuleDict({
            'main': PhysicsInformedOutput(
                self.d_model,
                self.output_dim,
                uncertainty=self.enable_uncertainty,
                bounds={'vp': (1500, 6000), 'vs': (500, 4000)}
            )
        })

class GravityExpert(BasePhysicsTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            physics_type="gravity",
            local_window=32,  # Larger window for smooth fields
            **kwargs
        )

    def _create_token_embedding(self):
        """Gravity-specific embedding with potential field awareness"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            self._create_gravity_processor(),
            nn.Linear(self.d_model*2, self.d_model)
        )

    def _create_gravity_processor(self):
        """Gravity-specific processing module"""
        return nn.Sequential(
            nn.Conv3d(self.d_model, self.d_model,
                     kernel_size=(5,5,5), padding='same', groups=self.d_model), # conv net to extract feature.
            nn.BatchNorm3d(self.d_model),
            nn.GELU()
        )

    def _create_output_heads(self):
        """Gravity-specific output with density constraints"""
        return nn.ModuleDict({
            'main': PhysicsInformedOutput(
                self.d_model,
                self.output_dim,
                uncertainty=self.enable_uncertainty,
                bounds={'density': (1.0, 5.0)}
            )
        })

class MagneticExpert(BasePhysicsTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            physics_type="magnetic",
            local_window=24,  # Medium window for magnetic features
            **kwargs
        )

    def _create_token_embedding(self):
        """Magnetic-specific embedding with spectral awareness"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            self._create_spectral_processor(),
            nn.LayerNorm(self.d_model)
        )

    def _create_spectral_processor(self):
        """Magnetic-specific spectral processing"""
        return nn.Sequential(
            SpectralFeatureExtractor(self.d_model), # SpectralFeatureExtractor to be added
            nn.Linear(self.d_model*2, self.d_model)
        )

    def _create_output_heads(self):
        """Magnetic-specific output with susceptibility constraints"""
        return nn.ModuleDict({
            'main': PhysicsInformedOutput(
                self.d_model,
                self.output_dim,
                uncertainty=self.enable_uncertainty,
                bounds={'susceptibility': (0.0, 0.1)}
            )
        })

class ElectricalExpert(BasePhysicsTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            physics_type="electrical",
            local_window=16,
            dropout=0.2,  # Higher dropout for EM noise
            **kwargs
        )

    def _create_token_embedding(self):
        """Electrical-specific embedding with resistivity awareness"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            self._create_resistivity_processor(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(0.2)
        )

    def _create_resistivity_processor(self):
        """Electrical-specific processing"""
        return nn.Sequential(
            ResistivityFeatureExtractor(self.d_model), # ResistivityFeatureExtractor
            nn.Linear(self.d_model*2, self.d_model)
        )

    def _create_output_heads(self):
        """Electrical-specific output with resistivity constraints"""
        return nn.ModuleDict({
            'main': PhysicsInformedOutput(
                self.d_model,
                self.output_dim,
                uncertainty=self.enable_uncertainty,
                bounds={'resistivity': (1e-6, 1e6)}
            )
        })
#@title Phase 2: Fusion expert warm-up with synthetic coupled data.
class MultiPhysicsFusion(nn.Module):
    def __init__(self, expert_models, hidden_dim=512):
        """
        Initialize fusion model with pre-trained experts

        Args:
            expert_models (dict): Dictionary of pre-trained expert models
                                {'seismic': model1, 'gravity': model2, ...}
            hidden_dim (int): Hidden dimension for fusion components
        """
        super().__init__()
        self.expert_names = ['seismic', 'gravity', 'magnetic', 'electrical']

        # Load pre-trained experts
        self.experts = nn.ModuleDict({
            name: expert_models[name] # '[name]' is this workable?
            for name in self.expert_names
        })

        # Freeze expert weights
        for expert in self.experts.values():
            for param in expert.parameters():
                param.requires_grad = False

        # Initialize fusion components
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, len(self.expert_names)), #map dim:(h_dim,4)
            nn.Softmax(dim=-1)
        )

        self.cross_physics_attn = nn.MultiheadAttention(
            hidden_dim, 8, batch_first=True
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Initialize fusion component weights
        self._init_fusion_weights()

    def _init_fusion_weights(self):
        """Initialize weights for fusion components"""
        for m in self.router.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, physics_inputs):
        """
        Forward pass through fusion model

        Args:
            physics_inputs (dict): Dictionary of physics inputs
                                 {'seismic': tensor1, 'gravity': tensor2, ...}
        """
        expert_outputs = []
        routing_weights = []

        # Process each physics modality
        for name in self.expert_names:
            with torch.no_grad():  # No gradients through experts
                out = self.experts[name](physics_inputs[name])
            expert_outputs.append(out)

            # Calculate routing weights (out shape: [batch_size, seq_len, hidden_dim])
            # mean over seq_len dimension to get [batch_size, hidden_dim]
            weights = self.router(out.mean(dim=1))
            routing_weights.append(weights) # [batch_size, hidden_dim]

        # Dynamic fusion
        routing_matrix = torch.stack(routing_weights, dim=1)  # [batch_size, num_experts, num_experts]
        fused_output = sum([out * w.unsqueeze(1).unsqueeze(-1)
                          for out, w in zip(expert_outputs, routing_weights)])

        # Cross-physics attention
        fused_output = self.cross_physics_attn(
            fused_output, fused_output, fused_output
        )[0]

        return self.predictor(fused_output)

def train_fusion_warmup(fusion_model, train_loader, val_loader,
                       num_epochs=10, learning_rate=1e-4):
    """
    Warm up fusion components using synthetic coupled data

    Args:
        fusion_model: MultiPhysicsFusion model
        train_loader: DataLoader for synthetic coupled training data
        val_loader: DataLoader for synthetic coupled validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for fusion components
    """
    # Only optimize fusion components
    optimizer = torch.optim.AdamW([
        {'params': fusion_model.router.parameters()},
        {'params': fusion_model.cross_physics_attn.parameters()},
        {'params': fusion_model.predictor.parameters()}
    ], lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        fusion_model.train()
        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Prepare input data
            physics_inputs = {
                name: batch[name] for name in fusion_model.expert_names
            }
            targets = batch['targets']  # Next timestep targets

            # Forward pass
            outputs = fusion_model(physics_inputs)
            loss = criterion(outputs, targets)

            # Backward pass (only updates fusion components)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        fusion_model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                physics_inputs = {
                    name: batch[name] for name in fusion_model.expert_names
                }
                targets = batch['targets']

                outputs = fusion_model(physics_inputs)
                val_loss += criterion(outputs, targets).item()

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.6f}")
        print(f"Val Loss: {val_loss/len(val_loader):.6f}")
        print("-------------------")

# Usage example:
def main():
    # Load pre-trained expert models
    expert_models = {
        'seismic': SeismicExpert(...),
        'gravity': GravityExpert(...),
        'magnetic': MagneticExpert(...),
        'electrical': ElectricalExpert(...)
    }

    # Initialize fusion model with pre-trained experts
    fusion_model = MultiPhysicsFusion(expert_models)

    # Create synthetic coupled data loaders
    train_loader = create_synthetic_dataloader(...)
    val_loader = create_synthetic_dataloader(...)

    # Warm up fusion components
    train_fusion_warmup(fusion_model, train_loader, val_loader)

#@title Phase 3: Full MoE fine-tuning with field data.

def fine_tune_moe(model, train_loader, val_loader,
                  num_epochs=20,
                  expert_lr=1e-5,    # Lower learning rate for experts
                  fusion_lr=1e-4,    # Higher learning rate for fusion
                  weight_decay=1e-4,
                  gradient_clip=1.0):
    """
    Full model fine-tuning with field data
    """
    # Separate parameter groups for different learning rates
    expert_params = []
    fusion_params = []

    for name, module in model.named_children():
        if name == 'experts':
            expert_params.extend(module.parameters())
        else:
            fusion_params.extend(module.parameters())

    optimizer = torch.optim.AdamW([
        {'params': expert_params, 'lr': expert_lr},
        {'params': fusion_params, 'lr': fusion_lr}
    ], weight_decay=weight_decay)

    # Reduce LR on plateau for both parameter groups
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    criterion = nn.MSELoss()

    # For early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        expert_original_outputs = {}  # Store original expert predictions

        for batch in train_loader:
            optimizer.zero_grad()

            physics_inputs = {
                name: batch[name] for name in model.expert_names
            }
            targets = batch['targets']

            # Store original expert predictions before update
            with torch.no_grad():
                for name, expert in model.experts.items():
                    expert_original_outputs[name] = expert(physics_inputs[name])

            # Forward pass
            outputs = model(physics_inputs)

            # Main prediction loss
            pred_loss = criterion(outputs, targets)

            # Expert stability regularization
            stability_loss = 0
            for name, expert in model.experts.items():
                current_output = expert(physics_inputs[name])
                stability_loss += criterion(current_output,
                                         expert_original_outputs[name])

            # Combined loss
            loss = pred_loss + 0.1 * stability_loss  # Adjust weight as needed

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                physics_inputs = {
                    name: batch[name] for name in model.expert_names
                }
                targets = batch['targets']
                outputs = model(physics_inputs)
                val_loss += criterion(outputs, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}")
        print("-------------------")

# Usage example:
def main():
    # Load model from Phase 2
    model = MultiPhysicsFusion.load_from_checkpoint('phase2_model.pth')

    # Create field data loaders
    train_loader = create_field_dataloader(...)
    val_loader = create_field_dataloader(...)

    # Fine-tune
    fine_tune_moe(model, train_loader, val_loader)