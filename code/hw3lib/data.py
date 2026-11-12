'Offline text preprocessing.'
from .runtime import np,torch
from pathlib import Path
import json,re,string
DATA=Path(__file__).with_name('data')
STOPWORDS={'a','an','the','on','in','of','with','and','at','is','to','there','are','photo','picture','show','me'}
PLURALS={'dogs':'dog','beaches':'beach','trees':'tree','boats':'boat','cats':'cat','cars':'car','bicycles':'bicycle','flowers':'flower','puppies':'puppy','kittens':'kitten'}
PARAPHRASE_CUES={'dog':'puppy','cat':'kitten','bicycle':'bike','car':'automobile','boat':'vessel','flower':'blossom','tree':'woodland','beach':'seaside','snow':'snowy','grass':'grassy','water':'waterside','sky':'sky'}
def retrieval_words(text):
    words=re.findall(r"[a-z]+",text.lower())
    return [PLURALS.get(w,w) for w in words if w not in STOPWORDS]
def canonical_tags(tags):
    'Sort and deduplicate tag sets. Preserve query order.'
    return ' '.join(sorted(set(str(t).lower() for t in tags)))
def catalog():return json.loads((DATA/'catalog.json').read_text())
def make_vocab(texts,word_fn):
    from collections import Counter
    counts=Counter(w for text in texts for w in word_fn(text))
    return ['','[UNK]']+sorted(counts,key=lambda w:(-counts[w],w))
def encode(texts,vocab,length,word_fn):
    ids={w:i for i,w in enumerate(vocab)};x=torch.zeros((len(texts),length),dtype=torch.long)
    for i,text in enumerate(texts):
        values=[ids.get(w,1) for w in word_fn(text)][:length]
        if values:x[i,:len(values)]=torch.tensor(values)
    return x

def retrieval_data():
    info=catalog();images=info['images'];keys=[canonical_tags(i['tags']) for i in images]
    rows=[]
    for image in images:
        tags=image['tags']
        for tag in tags:
            rows.append({'query':f'a photo of {tag}','required':[tag],'split':'train'})
            rows.append({'query':f'{PARAPHRASE_CUES[tag]} in a picture','required':[tag],'split':'train'})
        if len(tags)>=2:
            rows.append({'query':f'{tags[0]} with {tags[1]}','required':tags[:2],'split':'train'})
            rows.append({'query':f'{PARAPHRASE_CUES[tags[0]]} and {tags[1]}','required':tags[:2],'split':'demo'})
    # Supplied query paraphrases.
    rows.extend([{'query':'dogs on the beach','required':['dog','beach'],'split':'demo'},
                 {'query':'a puppy on the seaside','required':['dog','beach'],'split':'demo'}])
    train=[r for r in rows if r['split']=='train'];demo=[r for r in rows if r['split']=='demo']
    vocab=make_vocab(keys+[r['query'] for r in train],retrieval_words)
    key_tokens=encode(keys,vocab,20,retrieval_words)
    queries=encode([r['query'] for r in train],vocab,20,retrieval_words)
    relevant=torch.tensor([[set(r['required'])<=set(i['tags']) for i in images] for r in train],dtype=torch.bool)
    return {'catalog':info,'vocab':vocab,'key_tokens':key_tokens,'query_tokens':queries,'relevant':relevant,'train':train,'demo':demo}

def source_words(text):
    return text.lower().translate(str.maketrans('','',string.punctuation)).split()
def target_words(text):
    strip=string.punctuation.replace('[','').replace(']','')+'¿'
    return text.lower().translate(str.maketrans('','',strip)).split()
def translation_data():
    data=json.loads((DATA/'translation.json').read_text())
    train=data['train'];src_vocab=make_vocab([r[0] for r in train],source_words)
    tgt_vocab=make_vocab(['[start] '+r[1]+' [end]' for r in train],target_words)
    encoded={}
    for split in ['train','validation','test']:
        rows=data[split]
        src=encode([r[0] for r in rows],src_vocab,20,source_words)
        tgt=encode(['[start] '+r[1]+' [end]' for r in rows],tgt_vocab,21,target_words)
        encoded[split]=(src,tgt[:,:-1],tgt[:,1:])
    return {'raw':data,'source_vocab':src_vocab,'target_vocab':tgt_vocab,'encoded':encoded}
