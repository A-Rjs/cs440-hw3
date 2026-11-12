'Training and display helpers.'
from .runtime import torch,nn,np,configure
from .data import retrieval_data,translation_data,encode,retrieval_words,source_words
import time

def train_retrieval(student,epochs=12,seed=440):
    'Train query and tag representations using all relevant catalog keys as positives.'
    configure(seed);data=retrieval_data();model=student.TextRetriever(len(data['vocab']))
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    gen=torch.Generator().manual_seed(seed);losses=[];start=time.perf_counter()
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(data['query_tokens']),generator=gen);total=0.
        for batch in order.split(32):
            optimizer.zero_grad();q=model(data['query_tokens'][batch]);keys=model(data['key_tokens'])
            scores=(q@keys.T)/.15;positive=data['relevant'][batch]
            loss=(torch.logsumexp(scores,dim=1)-torch.logsumexp(scores.masked_fill(~positive,-torch.inf),dim=1)).mean()
            loss.backward();optimizer.step();total+=float(loss.detach())*len(batch)
        losses.append(total/len(order))
    model.eval()
    return model,data,{'loss':losses,'seconds':time.perf_counter()-start,'seed':seed,'epochs':epochs,'training_source':'Catalog tags and supplied paraphrases.'}

def search(student,model,data,query,k=5):
    model.eval()
    with torch.no_grad():
        q=model(encode([query],data['vocab'],20,retrieval_words));keys=model(data['key_tokens'])
        indices,scores=student.retrieve_topk(q,keys,[r['image_id'] for r in data['catalog']['images']],k)
    return indices,scores

def show_retrieval(student,model,data,query,k=5):
    import matplotlib.pyplot as plt
    from .data import DATA
    indices,scores=search(student,model,data,query,k)
    fig,axes=plt.subplots(1,k,figsize=(3*k,3),squeeze=False)
    for ax,index,score in zip(axes[0],np.asarray(indices)[0],np.asarray(scores)[0]):
        item=data['catalog']['images'][int(index)];ax.imshow(plt.imread(DATA/item['file']))
        ax.set_title(item['image_id']+'\n'+', '.join(item['tags'])+'\ncosine '+str(round(float(score),3)),fontsize=8);ax.axis('off')
    fig.suptitle(query);plt.tight_layout();plt.show()

class TranslationModel(nn.Module):
    'Encoder and decoder assembly with probability outputs.'
    def __init__(self,student,source_vocab_size,target_vocab_size,dim=32,heads=2,ff=64):
        super().__init__();self.student=student
        self.source_embedding=student.PositionalEmbedding(20,source_vocab_size,dim)
        self.target_embedding=student.PositionalEmbedding(20,target_vocab_size,dim)
        self.encoder=student.TransformerEncoder(dim,ff,heads)
        self.decoder=student.TransformerDecoder(dim,ff,heads)
        self.dropout=nn.Dropout(.1);self.output=nn.Linear(dim,target_vocab_size)
    def forward(self,source_tokens,decoder_tokens):
        enc_mask,self_mask,cross_mask=self.student.build_attention_masks(source_tokens,decoder_tokens)
        source=self.encoder(self.source_embedding(source_tokens),mask=enc_mask[:,0,:])
        decoded=self.decoder(self.target_embedding(decoder_tokens),source,self_mask=self_mask,cross_mask=cross_mask)
        return self.output(self.dropout(decoded)).softmax(dim=-1)

def train_translation(student,epochs=3,seed=440):
    configure(seed);data=translation_data()
    model=TranslationModel(student,len(data['source_vocab']),len(data['target_vocab']))
    optimizer=torch.optim.Adam(model.parameters(),lr=.002);gen=torch.Generator().manual_seed(seed)
    src,dec,tgt=data['encoded']['train'];history=[];start=time.perf_counter()
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(src),generator=gen);loss_sum=0.;correct_sum=0.;count=0
        for batch in order.split(64):
            optimizer.zero_grad();p=model(src[batch],dec[batch]);loss,acc=student.masked_token_metrics(tgt[batch],p)
            loss.backward();optimizer.step();n=int((tgt[batch]!=0).sum());loss_sum+=float(loss.detach())*n;correct_sum+=float(acc.detach())*n;count+=n
        model.eval()
        with torch.no_grad():
            vs,vd,vt=data['encoded']['validation'];ls=0.;ac=0.;nv=0
            for indices in torch.arange(len(vs)).split(64):
                loss,acc=student.masked_token_metrics(vt[indices],model(vs[indices],vd[indices]));n=int((vt[indices]!=0).sum());ls+=float(loss)*n;ac+=float(acc)*n;nv+=n
        history.append({'train_loss':loss_sum/count,'train_token_accuracy':correct_sum/count,'validation_loss':ls/nv,'validation_token_accuracy':ac/nv})
    model.eval()
    return model,data,{'history':history,'seconds':time.perf_counter()-start,'epochs':epochs,'seed':seed,'metric':'Token loss and accuracy with padding excluded.'}

def translate(student,model,data,text):
    source=encode([text],data['source_vocab'],20,source_words)
    vocab=data['target_vocab'];ids=student.greedy_decode(model,source,vocab.index('[start]'),vocab.index('[end]'),20)
    return ' '.join(vocab[i] for i in ids)
