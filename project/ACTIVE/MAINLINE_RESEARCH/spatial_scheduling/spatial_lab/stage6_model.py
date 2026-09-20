"""Batched causal encoders and values conditioned on executed numeric actions."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from .core import ContractError
from .feature_state import WIDTH

@dataclass
class ModelConfig:
    input_dim:int=WIDTH
    hidden_dim:int=384
    num_layers:int=6
    num_heads:int=8
    memory_dim:int=384
    dropout:float=0.
    num_roles:int=16
    schema:str='ain.stage6.numeric_conditioned_batched.v1'

def pack_graphs(graphs,device='cpu',adjacency=None):
    n=max(len(g.node_ids) for g in graphs);a=max(1,max(len(g.action_node_index) for g in graphs));b=len(graphs)
    x=torch.zeros(b,n,WIDTH);roles=torch.zeros(b,n,dtype=torch.long);types=roles.clone()
    valid=torch.zeros(b,n,dtype=torch.bool);adj=torch.eye(n,dtype=torch.bool).expand(b,n,n).clone()
    ai=torch.zeros(b,a,dtype=torch.long);av=torch.zeros(b,a,dtype=torch.bool)
    for k,g in enumerate(graphs):
        count=len(g.node_ids);x[k,:count]=g.node_features.cpu();roles[k,:count]=g.role_id.cpu()
        types[k,:count]=g.node_type.cpu();valid[k,:count]=True
        if adjacency is not None:adj[k,:count,:count]=adjacency[k]
        elif g.edge_index.numel():
            i,j=g.edge_index.cpu();adj[k,i,j]=True;adj[k,j,i]=True
        if adjacency is None and g.hyperedge_index.numel():
            ix=g.hyperedge_index.cpu()
            for h in ix[1].unique():
                nodes=ix[0,ix[1]==h];adj[k,nodes[:,None],nodes[None,:]]=True
        count=len(g.action_node_index);ai[k,:count]=g.action_node_index.cpu();av[k,:count]=True
    return {key:(v.pin_memory() if str(device).startswith('cuda') else v).to(device,non_blocking=True) for key,v in
            dict(x=x,roles=roles,types=types,valid=valid,adj=adj,action_index=ai,action_valid=av).items()}

class Block(nn.Module):
    def __init__(self,cfg,arm):
        super().__init__();h=cfg.hidden_dim;self.arm=arm;self.heads=cfg.num_heads
        self.norm=nn.LayerNorm(h);self.norm2=nn.LayerNorm(h)
        self.ff=nn.Sequential(nn.Linear(h,4*h),nn.GELU(),nn.Linear(4*h,h))
        if arm=='deep_sets':
            self.local=nn.Linear(h,h);self.context=nn.Linear(h,h)
        else:self.qkv=nn.Linear(h,3*h);self.output=nn.Linear(h,h)
    def forward(self,h,valid,adj):
        z=self.norm(h);mask=valid.unsqueeze(-1)
        if self.arm=='deep_sets':
            context=(z*mask).sum(1)/mask.sum(1).clamp_min(1)
            update=F.gelu(self.local(z)+self.context(context).unsqueeze(1))
        else:
            b,n,d=z.shape;q,k,v=self.qkv(z).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
            allowed=valid[:,None,None,:]
            if self.arm=='integrated_agent':allowed=allowed&adj[:,None,:,:]
            update=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0.)
            update=self.output(update.transpose(1,2).reshape(b,n,d))
        h=(h+update)*mask
        return (h+self.ff(self.norm2(h)))*mask

@dataclass
class Output:
    candidate_score:torch.Tensor
    node_embedding:torch.Tensor
    next_memory:torch.Tensor
    controls:torch.Tensor
    proxy:torch.Tensor
    value:torch.Tensor
    action_embedding:torch.Tensor
    context:torch.Tensor

class NumericPolicy(nn.Module):
    def __init__(self,cfg,arm):
        super().__init__();self.cfg=cfg;self.arm=arm;h=cfg.hidden_dim
        if cfg.hidden_dim!=cfg.memory_dim or h%cfg.num_heads:raise ContractError('STAGE6_MODEL_DIMENSIONS')
        self.input=nn.Sequential(nn.Linear(cfg.input_dim,h),nn.LayerNorm(h),nn.GELU())
        self.roles=nn.Embedding(cfg.num_roles,h);self.types=nn.Embedding(32,h)
        self.memory_projection=nn.Linear(h,h,bias=False)
        self.blocks=nn.ModuleList(Block(cfg,arm) for _ in range(cfg.num_layers))
        self.recurrent=nn.GRUCell(h,h)
        self.numeric_head=nn.Sequential(nn.Linear(h,h),nn.GELU(),nn.Linear(h,8))
        self.atom_head=nn.Linear(h,1)
        self.value_head=nn.Sequential(nn.Linear(h,h),nn.GELU(),nn.Linear(h,1))
        self.action_encoder=nn.Sequential(nn.Linear(h+16,h),nn.GELU(),nn.Linear(h,h))
        self.group_head=nn.Sequential(nn.Linear(3*h+1,2*h),nn.GELU(),nn.Linear(2*h,h),nn.GELU(),nn.Linear(h,2))
    @property
    def joint_schedule_decoder(self):return self.group_head
    def forward_batch(self,batch,memory=None):
        x=batch['x'];b,n,_=x.shape;h=self.cfg.hidden_dim
        if memory is None:memory=x.new_zeros(b,n,h)
        z=self.input(x)+self.roles(batch['roles'])+self.types(batch['types'])+self.memory_projection(memory)
        for layer in self.blocks:z=layer(z,batch['valid'],batch['adj'])
        nxt=self.recurrent(z.reshape(-1,h),memory.reshape(-1,h)).reshape(b,n,h)
        nxt=nxt*batch['valid'].unsqueeze(-1)
        ah=z.gather(1,batch['action_index'].unsqueeze(-1).expand(-1,-1,h))
        ah=ah*batch['action_valid'].unsqueeze(-1)
        context=(z*batch['valid'].unsqueeze(-1)).sum(1)/batch['valid'].sum(1,keepdim=True).clamp_min(1)
        atom=F.softplus(self.atom_head(ah).squeeze(-1))
        return Output(-atom,z,nxt,torch.sigmoid(self.numeric_head(ah)),atom,
                      F.softplus(self.value_head(context).squeeze(-1)),ah,context)
    def forward(self,G,memory=None):
        batch=pack_graphs([G],next(self.parameters()).device)
        out=self.forward_batch(batch,memory.unsqueeze(0) if memory is not None else None)
        a=len(G.action_node_index);n=len(G.node_ids)
        return Output(out.candidate_score[0,:a],out.node_embedding[0,:n],out.next_memory[0,:n],
                      out.controls[0,:a],out.proxy[0,:a],out.value[0],
                      out.action_embedding[0,:a],out.context[0])
    def group_values(self,out,indices,features,member_mask):
        """[B,C,K] semantic indices + [B,C,K,16] executed controls and masks."""
        ah=out.action_embedding;context=out.context
        if ah.ndim==2:ah=ah.unsqueeze(0);context=context.unsqueeze(0)
        b,c,k=indices.shape;h=ah.shape[-1]
        if ah.shape[1]==0:ah=ah.new_zeros(b,1,h)
        gathered=ah.gather(1,indices.clamp_min(0).reshape(b,-1).unsqueeze(-1).expand(-1,-1,h)).reshape(b,c,k,h)
        z=self.action_encoder(torch.cat([gathered,features],-1))*member_mask.unsqueeze(-1)
        count=member_mask.sum(-1,keepdim=True)
        pooled=torch.cat([z.sum(2),z.square().sum(2)/count.clamp_min(1),
                          context[:,None,:].expand(-1,c,-1),count/2],-1)
        return F.softplus(self.group_head(pooled))

def make(arm='integrated_agent',device='cpu',seed=541,config=None):
    torch.manual_seed(seed)
    if arm=='original_agent':arm='integrated_agent'
    if arm not in ('integrated_agent','deep_sets','set_transformer'):raise ContractError('UNKNOWN_BACKBONE')
    cfg=ModelConfig(**config) if config else ModelConfig()
    return NumericPolicy(cfg,arm).to(device),cfg

def numeric_features(schedule,view,config):
    from .actions import control_target
    values=[]
    for task in schedule:
        u,mask=control_target(task,view,config)
        values.append([float(a*b) for a,b in zip(u,mask)]+[float(v) for v in mask])
    return values

def align_memory(previous,old_ids,new_ids,width,device):
    """Differentiable semantic alignment; missing nodes start at zero."""
    n=max(map(len,new_ids));rows=[]
    for i,ids in enumerate(new_ids):
        index={key:j for j,key in enumerate(old_ids[i])} if previous is not None else {}
        zero=torch.zeros(width,device=device)
        vals=[previous[i,index[key]] if key in index else zero for key in ids]
        vals.extend(zero for _ in range(n-len(ids)));rows.append(torch.stack(vals))
    return torch.stack(rows)
