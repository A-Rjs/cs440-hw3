'Supplied attention module.'
from .runtime import torch,nn
import math
class BookAttention(nn.Module):
    'Attention with per head width D. Boolean True permits attention. Fully blocked rows return zero output and weights with finite gradients.'
    def __init__(self,embed_dim,num_heads):
        super().__init__();self.embed_dim=embed_dim;self.num_heads=num_heads
        self.query=nn.Linear(embed_dim,num_heads*embed_dim)
        self.key=nn.Linear(embed_dim,num_heads*embed_dim)
        self.value=nn.Linear(embed_dim,num_heads*embed_dim)
        self.output=nn.Linear(num_heads*embed_dim,embed_dim)
    def forward(self,query,key,value,attention_mask=None,return_attention_scores=False):
        b,t,d=query.shape;s=key.shape[1];h=self.num_heads
        q=self.query(query).reshape(b,t,h,d).transpose(1,2)
        k=self.key(key).reshape(b,s,h,d).transpose(1,2)
        v=self.value(value).reshape(b,s,h,d).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(d)
        if attention_mask is None:
            allowed=torch.ones((b,1,t,s),dtype=torch.bool,device=query.device)
        else:
            allowed=attention_mask[:,None,:,:].expand(b,1,t,s)
        has_key=allowed.any(dim=-1,keepdim=True)
        # Replace all-blocked rows BEFORE softmax, avoiding NaN backward.
        scores=scores.masked_fill(~allowed,float('-inf'))
        scores=torch.where(has_key,scores,torch.zeros_like(scores))
        weights=torch.softmax(scores,dim=-1).masked_fill(~allowed,0.)
        context=(weights@v).transpose(1,2).reshape(b,t,h*d)
        output=self.output(context)
        output=torch.where(has_key[:,0],output,torch.zeros_like(output))
        return (output,weights) if return_attention_scores else output
