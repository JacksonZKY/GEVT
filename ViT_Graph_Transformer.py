import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
import timm
import math
from transformers import ViTModel, ViTFeatureExtractor
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree  

class GraphModel(nn.Module):
    def __init__(self):
        super(GraphModel, self).__init__()
        self.vit0 = PretrainedViTModule()
        for param in self.vit0.parameters():
            param.requires_grad = False
        self.vit_lora_1 = My_LoRA_ViT_timm(r=16,alpha=32)
        self.vit_lora_2 = My_LoRA_ViT_timm(r=16,alpha=32)
        self.transformer = GraphTransformer(emb_size=768, num_layers=3, heads=8, forward_expansion=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_images, graph_masks, pad_masks):
        batch_size = batch_images.shape[0]
        num_in_batch = batch_images.shape[1]
        image_size = batch_images.shape[-1]
        images = batch_images.reshape(-1,3,image_size,image_size)

        with torch.no_grad():
            _, fixed_patch_embs = self.vit0(images)

        _, dist_patch_embs = self.vit_lora_1(images)
        direction_cls_embs, _ = self.vit_lora_2(images)

        fixed_patch_embs = fixed_patch_embs.reshape(batch_size,num_in_batch,-1,fixed_patch_embs.shape[-1])
        dist_patch_embs = dist_patch_embs.reshape(batch_size,num_in_batch,-1,dist_patch_embs.shape[-1])
        direction_cls_embs = direction_cls_embs.reshape(batch_size,num_in_batch,-1)
        
        dist_matrices = self.get_dist_matrix_from_patch_embs(fixed_patch_embs, dist_patch_embs, pad_masks)

        if self.training:
            graph_masks = (graph_masks+graph_masks.transpose(2,1))/2
        else:
            graph_masks = torch.zeros_like(dist_matrices)
            for batch in range(batch_size):
                dist_matrix = dist_matrices[batch]
                graph_masks[batch] = get_MST(dist_matrix)

        direct_matrices = self.transformer(direction_cls_embs, graph_masks, pad_masks)

        return dist_matrices, direct_matrices
    
    def get_dist_matrix_from_patch_embs(self, fixed_patch_embs, dist_patch_embs, pad_masks):
        batch_size = fixed_patch_embs.shape[0]
        pad_image_num = fixed_patch_embs.shape[1]
        device = fixed_patch_embs.device

        dist_matrix = torch.full((batch_size,pad_image_num,pad_image_num), float(1e20)).to(device)

        for batch in range(batch_size):
            pre_dist_embs = fixed_patch_embs[batch]
            trained_dist_embs = dist_patch_embs[batch]
            image_num = pad_masks[batch].sum().item()
            
            for i in range(image_num):
                for j in range(i+1,image_num):
                    pdist = nn.PairwiseDistance(p=2)   
                    trained_dist = pdist(trained_dist_embs[i],trained_dist_embs[j])

                    if self.training:
                        pre_dist = pdist(pre_dist_embs[i],pre_dist_embs[j])
                        softmax_pre_dist = torch.softmax(pre_dist,dim=-1)*pre_dist.shape[-1]
                        dist = torch.mean(trained_dist*(softmax_pre_dist))
                    else:
                        dist = torch.mean(trained_dist)

                    dist_matrix[batch][i][j] = dist
                    dist_matrix[batch][j][i] = dist

        return dist_matrix
    
    
class GraphTransformer(nn.Module):
    def __init__(self, emb_size=768, num_layers=3, heads=8, forward_expansion=4, dropout=0.01):
        super(GraphTransformer, self).__init__()
        self.srctransformer = MyTransformerEncoder(embed_size=emb_size, num_layers=num_layers, heads=heads, forward_expansion=forward_expansion, dropout=dropout)
        self.vir_src_emb = nn.Parameter(torch.randn(emb_size))
        self.vir_tar_emb = nn.Parameter(torch.randn(emb_size))

    def forward(self, embs, graph_masks, pad_masks):
        vir_src_embs = self.vir_src_emb.unsqueeze(0).unsqueeze(1).expand(embs.shape[0], 1, -1)
        vir_tar_embs = self.vir_tar_emb.unsqueeze(0).unsqueeze(1).expand(embs.shape[0], 1, -1)
        expanded_embs = torch.cat([vir_src_embs, vir_tar_embs, embs], dim=1)

        B,N,_ = graph_masks.shape
        expanded_graph_masks = torch.zeros(B,N+2,N+2)
        for batch_idx in range(graph_masks.shape[0]): 
            graph_masks[batch_idx].fill_diagonal_(torch.tensor(5.0))
        expanded_graph_masks[:,0:2,:] = 1
        expanded_graph_masks[:,2:N+2,2:N+2] = graph_masks
        expanded_graph_masks = expanded_graph_masks.to(embs.device)
        result_embs = self.srctransformer(expanded_embs, graph_masks=expanded_graph_masks, pad_masks=pad_masks)  
        A_matrix = self.cls_direction(result_embs)
        
        return A_matrix
    
    def cls_direction(self, embs):
        B,N,D = embs.shape[0],embs.shape[1]-2,embs.shape[2]
        src_cls, tar_cls, embs = embs[:,0],embs[:,1],embs[:,2:]
        anchor = (src_cls - tar_cls).view(B,D)
        anchor_norm = anchor / anchor.norm(dim=-1, keepdim=True)
        src_embs = embs.unsqueeze(dim=2).repeat(1,1,N,1)
        tar_embs = embs.unsqueeze(dim=1).repeat(1,N,1,1)
        direct_embs = src_embs - tar_embs
        direct_embs_flat = direct_embs.reshape(B, N * N, D)
        dot_products = torch.bmm(direct_embs_flat, anchor_norm.unsqueeze(dim=2))  # Resulting shape (B, N*N, 1)
        # Calculate the magnitude of projection
        projection_lengths = dot_products.view(B,N,N)
        A_matrix = projection_lengths - projection_lengths.transpose(2,1)
        return A_matrix
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, pad_mask, graph_mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
       
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if graph_mask is not None:
            graph_mask = graph_mask.reshape(N, 1, query_len, key_len).repeat(1,self.heads,1,1)
            energy = energy*graph_mask
        
        if pad_mask is not None:
            float_mask  = torch.where(pad_mask, torch.tensor(1.0), torch.tensor(0.0)).unsqueeze(2)
            float_mask = torch.bmm(float_mask, float_mask.transpose(2,1))
            float_mask = float_mask.unsqueeze(1).repeat(1,self.heads,1,1)
            energy = energy.masked_fill(float_mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention.float(), values.float()]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, pad_mask, graph_mask):
        attention = self.attention(value, key, query, pad_mask, graph_mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
 
class MyTransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout):
        super(MyTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
    def forward(self, x, graph_masks, pad_masks=None):
        B, T, E = x.size()
        if pad_masks is not None:
            pad_masks = torch.cat([torch.ones(B, 2, device=pad_masks.device, dtype=torch.bool), pad_masks], dim=1)  # update mask for CLS tokens
        out = x
        for layer in self.layers:
            out = layer(out, out, out, pad_masks, graph_masks)
        return out

class PretrainedViTModule(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224'):
        super(PretrainedViTModule, self).__init__()
        self.vit_model = ViTModel.from_pretrained(model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        for name, param in self.vit_model.named_parameters():
            param.requires_grad = False

    def forward(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.vit_model.device) for k, v in inputs.items()}    
        outputs = self.vit_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        cls_token = last_hidden_states[:, 0, :]
        patch_tokens = last_hidden_states[:, 1:, :]
        return cls_token, patch_tokens

class My_LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model_type= "base", r=16, alpha=32):
        super(My_LoRA_ViT_timm, self).__init__()
        self.weightInfo={
        # "small":"WinKawaks/vit-small-patch16-224",
        "base":"vit_base_patch16_224",
        "base_dino":"vit_base_patch16_224.dino", # 21k -> 1k
        "base_sam":"vit_base_patch16_224.sam", # 1k
        "base_mill":"vit_base_patch16_224_miil.in21k_ft_in1k", # 1k
        "base_beit":"beitv2_base_patch16_224.in1k_ft_in22k_in1k",
        "base_clip":"vit_base_patch16_clip_224.laion2b_ft_in1k", # 1k
        "base_deit":"deit_base_distilled_patch16_224", # 1k
        "large":"google/vit-large-patch16-224",
        "large_clip":"vit_large_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
        "large_beit":"beitv2_large_patch16_224.in1k_ft_in22k_in1k", 
        "huge_clip":"vit_huge_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
        "giant_eva":"eva_giant_patch14_224.clip_ft_in1k", # laion-> 1k
        "giant_clip":"vit_giant_patch14_clip_224.laion2b",
        "giga_clip":"vit_gigantic_patch14_clip_224.laion2b"
        }
        self.model = timm.create_model(self.weightInfo[vit_model_type], pretrained=True)
        self.lora_model = LoRA_ViT_timm(vit_model=self.model, r=r, alpha=alpha)
    def forward(self, x):
        return self.lora_model(x)
        
class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, alpha: int, lora_layer=None):
        super(LoRA_ViT_timm, self).__init__()

        assert r > 0
        #assert alpha > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))
        self.w_As = [] 
        self.w_Bs = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                r,
                alpha
            )
        self.reset_parameters()
        self.lora_vit = vit_model
        self.lora_vit.pos_drop.p = 0.0

        for blk in self.lora_vit.blocks:
            if hasattr(blk.attn, 'dropout'):
                blk.attn.dropout.p = 0.0
            if hasattr(blk.mlp, 'dropout'):
                blk.mlp.dropout.p = 0.0

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lora_vit.patch_embed(x)

        cls_tokens = self.lora_vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.lora_vit.pos_embed
        x = self.lora_vit.pos_drop(x)

        for blk in self.lora_vit.blocks:
            x = blk(x)
        return x[:, 0], x[:, 1:]  # CLS token, Other tokens


class _LoRA_qkv_timm(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        r: int,
        alpha: int
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += (self.alpha // self.r) * new_q
        qkv[:, :, -self.dim :] += (self.alpha // self.r) * new_v
        return qkv
    
def get_MST(dist_matrix):       

    dist_matrix_numpy = dist_matrix.cpu().detach().numpy()
    # Convert to a sparse CSR matrix
    sparse_matrix = csr_matrix(dist_matrix_numpy)

    # Compute the MST
    mst_matrix_sparse = minimum_spanning_tree(sparse_matrix)

    # Convert the resulting MST back to a dense format
    mst_matrix = mst_matrix_sparse.toarray()

    # Make the matrix symmetric
    mst_matrix = np.maximum(mst_matrix, mst_matrix.T)

    # Convert back to PyTorch tensor
    mst_tensor = torch.from_numpy(mst_matrix).to(dist_matrix.device)
    mst_tensor = torch.where(mst_tensor>0,1,0)

    return mst_tensor
    
