'Check saved code and answer formats.'
from .runtime import torch,np,configure

def run(s,a,part='all'):
    configure();results=[]
    def test(name,fn):
        try:fn();print('PASS',name);results.append(True)
        except Exception as e:print('CHECK',name,':',type(e).__name__,str(e));results.append(False)
    def positions():
        m=s.PositionalEmbedding(8,12,4);x=torch.tensor([[2,1,0]])
        assert m(x).shape==(1,3,4),'Expected B,T,D embedding shape'
        assert m.compute_mask(x).dtype==torch.bool,'Mask must be boolean'
    def encoder():
        m=s.TransformerEncoder(4,8,2);assert m(torch.randn(2,3,4),mask=torch.ones(2,3,dtype=torch.bool)).shape==(2,3,4)
    def retriever():
        m=s.TextRetriever(12,embed_dim=4,dense_dim=8);x=torch.tensor([[2,1,0],[0,0,0]])
        z=m(x);assert z.shape==(2,4) and torch.isfinite(z).all(),'Finite B,D embeddings required'
    def ranking():
        indices,scores=s.retrieve_topk(np.eye(2),np.eye(2),['b','a'],1)
        assert np.asarray(indices).shape==np.asarray(scores).shape==(2,1)
    def masks():
        masks=s.build_attention_masks(torch.tensor([[2,0,3]]),torch.tensor([[4,5]]))
        assert [tuple(m.shape) for m in masks]==[(1,1,3),(1,2,2),(1,2,3)]
        assert all(m.dtype==torch.bool for m in masks)
    def decoder():
        m=s.TransformerDecoder(4,8,2);assert m(torch.randn(1,2,4),torch.randn(1,3,4)).shape==(1,2,4)
    def metrics():
        p=torch.full((1,2,5),.2,requires_grad=True);loss,acc=s.masked_token_metrics(torch.tensor([[2,0]]),p)
        assert isinstance(loss,torch.Tensor) and loss.ndim==0,'Loss must be a scalar tensor'
        assert isinstance(acc,torch.Tensor) and acc.ndim==0,'Accuracy must be a scalar tensor'
        loss.backward();assert p.grad is not None,'Loss must keep gradients'
    def greedy():
        class Stop(torch.nn.Module):
            def forward(self,source,decoder):
                p=torch.zeros(1,decoder.shape[1],5);p[:,:,2]=1;return p
        result=s.greedy_decode(Stop(),torch.tensor([[3,0]]),3,2,4)
        assert isinstance(result,list) and all(isinstance(x,int) for x in result)
    def objectives():
        assert isinstance(a.Q1_FIGURE_SCORES,(list,tuple)) and len(a.Q1_FIGURE_SCORES)==3,'Q1_FIGURE_SCORES must have 3 numeric entries'
        assert all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in a.Q1_FIGURE_SCORES),'Scores must be numbers'
        assert isinstance(a.Q1_TIED_IDS,(list,tuple)) and len(a.Q1_TIED_IDS)==2 and all(isinstance(x,str) for x in a.Q1_TIED_IDS),'Two string IDs required'
        for name in ['Q1_PROBABILITIES','Q1_PAYLOAD','Q1_SYNONYM','Q2_MASK','Q2_ACCURACY','Q2_POSITIONS']:
            value=getattr(a,name);assert isinstance(value,str) and value.strip().upper() in 'ABCD' and len(value.strip())==1,f'{name}: choose one letter A-D'
        assert isinstance(a.Q2_SHIFT,(list,tuple)) and len(a.Q2_SHIFT)==4 and all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in a.Q2_SHIFT),'Q2_SHIFT must have 4 numeric IDs'
    groups={'q1':[('positions',positions),('encoder',encoder),('retriever',retriever),('ranking',ranking)],'q2':[('masks',masks),('decoder',decoder),('metrics',metrics),('greedy',greedy)],'answers':[('Answer formats',objectives)]}
    if part not in ['all',*groups]:raise ValueError('part must be all, q1, q2 or answers')
    for group,tests in groups.items():
        if part in ('all',group):
            for name,fn in tests:test(name,fn)
    print('Format checks finished.')
    return all(results)
