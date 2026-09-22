"""Fixed research WER and paired group uncertainty, independent of predictions."""
import re
import unicodedata
import numpy as np
from jiwer import process_words


def normalize(text):
    text = unicodedata.normalize('NFKC',text).lower().replace("'",'').replace('’','')
    return ' '.join(re.sub(r'[^a-z0-9\s]',' ',text).split())


def score(records,prediction):
    assert isinstance(prediction,list) and len(prediction)==len(records)
    rows=[]
    for record,text in zip(records,prediction):
        assert isinstance(text,str)
        reference,hypothesis=normalize(record['text']),normalize(text)
        assert reference
        result=process_words(reference,hypothesis)
        rows.append({'group':record['group'],'reference_words':len(reference.split()),
            'errors':result.substitutions+result.deletions+result.insertions})
    words=sum(r['reference_words'] for r in rows)
    return {'wer':sum(r['errors'] for r in rows)/words,'words':words,'rows':rows}


def compare(baseline,candidate,seed=20260921):
    assert len(baseline['rows'])==len(candidate['rows'])
    groups={}
    for a,b in zip(baseline['rows'],candidate['rows']):
        assert (a['group'],a['reference_words'])==(b['group'],b['reference_words'])
        groups.setdefault(a['group'],[0,0,0])
        groups[a['group']][0]+=a['errors']
        groups[a['group']][1]+=b['errors']
        groups[a['group']][2]+=a['reference_words']
    values=np.asarray(list(groups.values()),dtype=float)
    rng=np.random.default_rng(seed)
    sums=values[rng.integers(0,len(values),size=(5000,len(values)))].sum(axis=1)
    differences=(sums[:,1]-sums[:,0])/sums[:,2]
    upper=float(np.quantile(differences,.975))
    gain=1-candidate['wer']/baseline['wer'] if baseline['wer']>0 else 0
    return {'relative_wer_gain':gain,'candidate_wer':candidate['wer'],'baseline_wer':baseline['wer'],
        'groups':len(values),'paired_group_97_5_upper_difference':upper,
        'qualified':bool(gain>=.1 and len(values)>=2 and upper<0),
        'uncertainty_note':'Descriptive paired group bootstrap; fresh confirmation groups per attempt; not a finite-sample error guarantee.'}
