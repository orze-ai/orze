import numpy as np, math
from sklearn.ensemble import HistGradientBoostingRegressor

HGB=dict(max_iter=250,learning_rate=0.08,max_leaf_nodes=31,l2_regularization=1)

def _hgb(X,y,seed):
    m=HistGradientBoostingRegressor(random_state=seed,**HGB)
    m.fit(X,y)
    return m

def _trend_slope(X,y,fn):
    ed=X[:,fn.index('elapsed_day')]
    cols=[np.ones(len(y)),ed]
    for name in ['hr','mnth','weathersit','weekday']:
        v=X[:,fn.index(name)]
        for k in np.unique(v)[1:]:
            cols.append((v==k).astype(float))
    for name in ['workingday','holiday','temp','hum','windspeed']:
        cols.append(X[:,fn.index(name)])
    A=np.column_stack(cols)
    coef=np.linalg.lstsq(A,np.log1p(y),rcond=None)[0]
    return float(coef[1])

def _branch(b,Xtr,ytr,Xte,fn,seed):
    info={}
    if b=='raw':
        p=_hgb(Xtr,ytr,seed).predict(Xte)
    elif b=='log':
        p=np.expm1(_hgb(Xtr,np.log1p(ytr),seed).predict(Xte))
    elif b=='log_trend':
        i=fn.index('elapsed_day'); iy=fn.index('yr')
        slope=_trend_slope(Xtr,ytr,fn)
        ref=float(Xtr[:,i].max())
        z=np.log1p(ytr)-slope*(Xtr[:,i]-ref)
        keep=[j for j in range(Xtr.shape[1]) if j not in (i,iy)]
        m=_hgb(Xtr[:,keep],z,seed)
        p=np.expm1(m.predict(Xte[:,keep])+slope*(Xte[:,i]-ref))
        info={'slope_per_day':slope,'annual_growth_factor':math.exp(365.0*slope),'ref_day':ref,'dropped_features':['elapsed_day','yr']}
    else:
        raise ValueError(b)
    return np.maximum(np.asarray(p,dtype=float),0.0),info

def fit_predict(train,inputs,seed):
    fn=list(train['feature_names'])
    X=np.asarray(train['X'],dtype=float); y=np.asarray(train['y'],dtype=float)
    Xte=np.asarray(inputs['X'],dtype=float)
    i=fn.index('elapsed_day')
    days=np.sort(np.unique(X[:,i]))
    branches=['raw','log','log_trend']
    scores={b:[] for b in branches}; errors={}; fold_info=[]
    for k in (1,2):
        lo=len(days)-61*k; hi=len(days)-61*(k-1)
        mte=np.isin(X[:,i],days[lo:hi]); mtr=X[:,i]<days[lo]
        fold_info.append({'fold':k,'train_days':int(len(np.unique(X[mtr,i]))),'holdout_day_range':[float(days[lo]),float(days[hi-1])],'holdout_rows':int(mte.sum())})
        for b in branches:
            try:
                p,_=_branch(b,X[mtr],y[mtr],X[mte],fn,seed)
                scores[b].append(float(np.sqrt(np.mean((p-y[mte])**2))))
            except Exception as exc:
                errors['%s_fold%d'%(b,k)]=type(exc).__name__
                scores[b].append(float('inf'))
    mean_scores={b:float(np.mean(scores[b])) for b in branches}
    selected=min(branches,key=lambda b:(mean_scores[b],branches.index(b)))
    fallback=None
    try:
        p,info=_branch(selected,X,y,Xte,fn,seed)
    except Exception as exc:
        fallback='raw_after_%s'%type(exc).__name__
        p,info=_branch('raw',X,y,Xte,fn,seed)
    nonfinite=int((~np.isfinite(p)).sum())
    if nonfinite:
        p=np.nan_to_num(p,nan=float(np.median(y)),posinf=float(y.max()),neginf=0.0)
    findings={'selected_branch':selected,'forward_fold_rmse':scores,'mean_forward_rmse':mean_scores,'folds':fold_info,'errors':errors,'fallback':fallback,'nonfinite_replaced':nonfinite,'final_info':info,'train_days_total':int(len(days)),'train_yr_share':float(np.mean(X[:,fn.index('yr')])),'note':'selection uses only training dates; no evaluation covariates used beyond prediction'}
    return {'prediction':p.tolist(),'findings':findings}
