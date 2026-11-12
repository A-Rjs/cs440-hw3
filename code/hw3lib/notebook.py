'Load saved notebook definitions.'
import json,types,sys
from pathlib import Path

def load_submission(path, part='all'):
    'Load marked definitions and answers from a saved notebook.'
    notebook=json.loads(Path(path).read_text(encoding='utf-8'))
    module=types.ModuleType('student');objective=types.ModuleType('answers')
    exec('from hw3lib.runtime import torch, nn, np\nfrom hw3lib.attention import BookAttention',module.__dict__)
    module.__load_errors__=[];objective.__load_errors__=[]
    found={'code':0,'answers':0}
    for i,cell in enumerate(notebook.get('cells',[])):
        source=cell.get('source','');source=''.join(source) if isinstance(source,list) else source
        if cell.get('cell_type')!='code':continue
        kind='code' if source.startswith('# GRADED: code') else 'answers' if source.startswith('# GRADED: answers') else None
        if kind is None or part not in ('all',kind):continue
        found[kind]+=1;target=module if kind=='code' else objective
        try:exec(compile(source,str(path)+':cell'+str(i),'exec'),target.__dict__)
        except Exception as exc:target.__load_errors__.append(f'cell {i}: {type(exc).__name__}: {exc}')
    for kind,target in [('code',module),('answers',objective)]:
        if part in ('all',kind) and not found[kind]:target.__load_errors__.append('Missing # GRADED: '+kind+' marker cells')
    return module,objective

def check(part='all',notebook_path='hw3.ipynb'):
    'Save the notebook before checking code and answer formats.'
    from .public_checks import run
    try:
        student,answers=load_submission(notebook_path)
        for error in student.__load_errors__+answers.__load_errors__:print('Definition cell error:',error)
        return run(student,answers,part)
    except Exception as exc:
        print(f'Cannot check saved notebook: {type(exc).__name__}: {exc}')
        print('Save hw3.ipynb and try again.')
        return False
