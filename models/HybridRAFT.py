import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Retrieval import RetrievalTool

# =============================================================================
# 1. Event Contextualizer (Enhanced TimeXer)
# =============================================================================
class EventContextualizer(nn.Module):
    """
    Enhanced Event Contextualizer:
    Uses embeddings for categorical time features and a Gated Residual Network (GRN)
    to better capture time dependencies.
    """
    def __init__(self, d_model):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x_mark):
        x = self.embedding(x_mark)
        g = self.gate(x)
        return x * g

# =============================================================================
# 2. Periodic Profiler (Enhanced TimesNet with Inception)
# =============================================================================
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(InceptionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class PeriodicProfiler(nn.Module):
    """
    Enhanced Periodic Profiler:
    Uses Inception Blocks to capture multi-scale temporal patterns within periods.
    """
    def __init__(self, seq_len, k=2, d_model=32):
        super().__init__()
        self.seq_len = seq_len
        self.k = k
        # Use InceptionBlock instead of simple 1x1 Conv
        # Input channels = 1 (since we process each time series independently in the 2D space)
        # Output channels = 1
        self.inception = InceptionBlock(1, 1, num_kernels=4)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        B, T, C = x.shape
        
        # 1. FFT
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, self.k)
        period_list = top_list.detach().cpu().numpy()
        
        res = []
        for period in period_list:
            if period == 0: period = 1
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([B, (length - self.seq_len), C]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            
            # Reshape: [B, Length//Period, Period, C] -> [B*C, 1, Length//Period, Period]
            out = out.permute(0, 2, 1).reshape(B * C, 1, length // period, period)
            
            # Inception Processing
            out = self.inception(out) # [B*C, 1, H, W]
            
            # Reshape back
            out = out.reshape(B, C, -1).permute(0, 2, 1)
            res.append(out[:, :self.seq_len, :])
            
        res = torch.stack(res, dim=-1).sum(-1)
        return res

# =============================================================================
# 3. Dynamic Graph Relation Encoder (Enhanced with Dynamic Adjacency)
# =============================================================================
class DynamicGraphEncoder(nn.Module):
    """
    Enhanced Dynamic Graph Encoder:
    Computes a dynamic adjacency matrix based on the input features + static node embeddings.
    """
    def __init__(self, num_nodes, d_model):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        # Static Node Embeddings
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, d_model))
        
        # Dynamic Adjacency Generators
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        
        # Learnable alpha for fusion
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Feature Extractor for Dynamic Graph (Pool over time)
        self.feature_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_proj = nn.Linear(1, d_model) # Project input scalar to d_model if needed, or just use node_embedding

    def forward(self, x):
        # x: [Batch, Seq_Len, Num_Nodes]
        B, T, N = x.shape
        
        # 1. Compute Dynamic Adjacency
        # We want to capture correlations between nodes based on current input x.
        # Simple approach: Correlation of time series.
        # Or: Use node embeddings modulated by input.
        
        # Let's use a lightweight attention on Node Embeddings, 
        # but we can make it dynamic by adding a "Global Context" from x.
        
        # Global Context: [B, N] (Mean over time)
        global_ctx = x.mean(dim=1, keepdim=True) # [B, 1, N]
        # This is just a scalar per node. Not enough for "content based" attention.
        
        # Better: Use the static node embedding as the base, 
        # and learn a dynamic attention weight.
        
        # Static Adjacency (Global Structure)
        adj_static = F.softmax(F.relu(torch.mm(self.node_embedding, self.node_embedding.transpose(0, 1))), dim=1)
        
        # Dynamic Adjacency (Local Correlations)
        # [B, T, N] -> [B, N, T]
        x_t = x.permute(0, 2, 1)
        # Compute correlation matrix for each batch: [B, N, N]
        # Normalize x_t
        x_t_norm = F.normalize(x_t, dim=2)
        adj_dyn = torch.bmm(x_t_norm, x_t_norm.transpose(1, 2)) # [B, N, N]
        adj_dyn = F.softmax(F.relu(adj_dyn), dim=2)
        
        # Fuse Adjacencies
        # alpha = 0.5 # Hyperparameter or learnable
        adj = self.alpha * adj_static + (1 - self.alpha) * adj_dyn
        
        # 2. Graph Convolution
        # X' = A * X
        # [B, N, N] * [B, N, T] -> [B, N, T]
        out = torch.bmm(adj, x_t)
        
        out = out.permute(0, 2, 1) # [B, T, N]
        
        return out, adj

# =============================================================================
# 4. HybridRAFT (Main Model)
# =============================================================================
class Model(nn.Module):
    """
    HybridRAFT: 主-辅赋能架构 (Enhanced)
    Core: RAFT
    Auxiliary: EventContextualizer, PeriodicProfiler, DynamicGraphEncoder
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.device = torch.device(f'cuda:{configs.gpu}')
        
        # --- Auxiliary Modules ---
        # 1. Event Contextualizer
        self.event_encoder = EventContextualizer(self.channels)
        
        # 2. Periodic Profiler
        self.periodic_profiler = PeriodicProfiler(self.seq_len, k=5)
        
        # 3. Dynamic Graph Encoder
        self.graph_encoder = DynamicGraphEncoder(num_nodes=self.channels, d_model=32)
        
        # --- RAFT Backbone ---
        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        
        self.n_period = configs.n_period
        self.topm = configs.topm
        
        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
        )
        
        self.period_num = self.rt.period_num[-1 * self.n_period:]
        
        module_list = [
            nn.Linear(self.pred_len // g, self.pred_len)
            for g in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)
        
        # Final Fusion
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)
        
        # Spatial Injection Layer for RAFT
        self.spatial_gate = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.Sigmoid()
        )

    def prepare_dataset(self, train_data, valid_data, test_data):
        self.rt.prepare_dataset(train_data)
        
        self.retrieval_dict = {}
        
        print('Doing Train Retrieval')
        train_rt = self.rt.retrieve_all(train_data, train=True, device=self.device)

        print('Doing Valid Retrieval')
        valid_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device)

        print('Doing Test Retrieval')
        test_rt = self.rt.retrieve_all(test_data, train=False, device=self.device)

        del self.rt
        torch.cuda.empty_cache()
            
        self.retrieval_dict['train'] = train_rt.detach()
        self.retrieval_dict['valid'] = valid_rt.detach()
        self.retrieval_dict['test'] = test_rt.detach()

    def forward(self, x, index, x_mark, mode='train'):
        # x: [Batch, Seq_Len, Channels]
        # index: [Batch]
        # x_mark: [Batch, Seq_Len, 4]
        
        bsz, seq_len, channels = x.shape
        index = index.to(self.device)
        
        # --- 1. Signal Enhancement Pipeline ---
        # Event Context
        event_ctx = self.event_encoder(x_mark) # [B, T, C]
        
        # Periodic Features
        periodic_feat = self.periodic_profiler(x) # [B, T, C]
        
        # Fusion: Original + Event + Periodic
        x_enhanced = x + event_ctx + periodic_feat
        
        # --- 2. Spatial Relation Extraction ---
        # Extract dynamic spatial structure from the current batch
        # spatial_ctx: [B, T, C], adj: [B, N, N] (Dynamic)
        spatial_ctx, adj = self.graph_encoder(x_enhanced)
        
        # --- 3. RAFT Backbone Execution ---
        
        # A. Direct Prediction Path (Linear)
        x_offset = x_enhanced[:, -1:, :].detach()
        x_norm = x_enhanced - x_offset
        pred_direct = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
        
        # B. Retrieval Path
        pred_from_retrieval = self.retrieval_dict[mode][:, index.cpu()] 
        pred_from_retrieval = pred_from_retrieval.to(self.device)
        
        retrieval_pred_list = []
        
        for i, pr in enumerate(pred_from_retrieval):
            g = self.period_num[i]
            
            pr = pr.reshape(bsz, self.pred_len // g, g, channels)
            pr = pr[:, :, 0, :] 
            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1) # [B, P, C]
            
            # --- Spatial Injection ---
            # pr: [B, P, C] -> [B, C, P]
            pr_in = pr.permute(0, 2, 1)
            
            # adj: [B, N, N]
            # Einsum: bnm, bmp -> bnp
            pr_spatial = torch.einsum('bnm, bmp -> bnp', adj, pr_in)
            pr_spatial = pr_spatial.permute(0, 2, 1) # [B, P, C]
            
            gate = self.spatial_gate(pr)
            pr = pr * (1 - gate) + pr_spatial * gate
            
            retrieval_pred_list.append(pr)
            
        pred_retrieval_sum = torch.sum(torch.stack(retrieval_pred_list), dim=0) # [B, P, C]
        
        # C. Final Combination
        combined = torch.cat([pred_direct, pred_retrieval_sum], dim=1) # [B, 2*P, C]
        combined = combined.permute(0, 2, 1)
        
        final_pred = self.linear_pred(combined) # [B, C, P]
        final_pred = final_pred.permute(0, 2, 1) # [B, P, C]
        
        final_pred = final_pred + x_offset
        
        return final_pred
