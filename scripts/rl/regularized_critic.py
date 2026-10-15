"""D251 flight-balanced ridge critic. No actor updates, reward edits or test-set tuning."""
import numpy as np
import torch

def fit_ridge(features, targets, groups, alpha):
    x=np.asarray(features,dtype=np.float64);y=np.asarray(targets,dtype=np.float64);g=np.asarray(groups)
    if x.ndim!=2 or y.shape!=(len(x),) or len(g)!=len(x) or not len(x) or alpha<=0:
        raise ValueError('invalid critic fit inputs')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite fit data')
    kinds=np.unique(g);weights=np.array([1./(len(kinds)*np.sum(g==a)) for a in g])
    scale=np.sqrt(x.shape[1]);x=x/scale
    mx=(weights[:,None]*x).sum(0);my=float(weights@y)
    z=(x-mx)*np.sqrt(weights[:,None]);t=(y-my)*np.sqrt(weights)
    dual=np.linalg.solve(z@z.T+float(alpha)*np.eye(len(x)),t)
    w=z.T@dual;bias=my-float(mx@w)
    head=torch.nn.Linear(x.shape[1],1)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy((w/scale)[None,:]).float());head.bias.fill_(bias)
    return head

def select_ridge(features,targets,groups,alphas):
    x=np.asarray(features);y=np.asarray(targets);g=np.asarray(groups);rows=[]
    if len(np.unique(g))<3:raise ValueError('whole-flight tuning requires at least three development flights')
    for alpha in alphas:
        folds=[]
        for held in np.unique(g):
            train=g!=held;val=~train
            h=fit_ridge(x[train],y[train],g[train],alpha)
            with torch.no_grad():v=h(torch.as_tensor(x[val],dtype=torch.float32)).flatten().numpy()
            folds.append(dict(heldout_group=str(held),training_groups=[str(a) for a in np.unique(g[train])],mse=float(np.mean((v-y[val])**2))))
        rows.append(dict(alpha=float(alpha),mse=float(np.mean([f['mse'] for f in folds])),folds=folds))
    best=min(rows,key=lambda a:(a['mse'],-a['alpha']))
    return fit_ridge(x,y,g,best['alpha']),dict(selected_alpha=best['alpha'],validation_mse=best['mse'],candidates=rows)
