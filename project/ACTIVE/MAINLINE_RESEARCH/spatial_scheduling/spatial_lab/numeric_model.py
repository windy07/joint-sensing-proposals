"""Device-valued proposal heads on the retained three representations."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import torch
from torch import nn

@dataclass
class NumericOutput:
 candidate_score:torch.Tensor
 node_embedding:torch.Tensor
 next_memory:torch.Tensor
 controls:torch.Tensor
 proxy:torch.Tensor
 value:torch.Tensor

class NumericPolicy(nn.Module):
 def __init__(self,backbone,width=48):
  super().__init__();self.backbone=backbone
  self.numeric_head=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,8))
  self.proxy_head=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))
  self.value_head=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))
 @property
 def joint_schedule_decoder(self):return self.backbone.joint_schedule_decoder
 def forward(self,G,memory=None):
  out=self.backbone(G,memory);h=out.node_embedding;ah=h[G.action_node_index]
  controls=torch.sigmoid(self.numeric_head(ah))
  return NumericOutput(out.candidate_score,h,out.next_memory,controls,
                        self.proxy_head(ah).squeeze(-1),self.value_head(h.mean(0)).squeeze(-1))
