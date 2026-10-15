"""D253 state-aware local critic, with preprocessing learned on training folds only."""
import numpy as np
import torch

class StateCritic(torch.nn.Module):
    def __init__(self,hidden,state_dim):
        super().__init__();self.hidden=hidden;self.state_dim=state_dim
        self.register_buffer('state_mean',torch.zeros(state_dim))
        self.register_buffer('state_scale',torch.ones(state_dim))
        self.register_buffer('state_weight',torch.tensor(1.))
        self.head=torch.nn.Linear(hidden+state_dim,1)
    def forward(self,features,state):
        if features.shape[-1]!=self.hidden or state.shape[-1]!=self.state_dim:
            raise ValueError('critic feature schema mismatch')
        a=features.detach().float()/np.sqrt(self.hidden)
        b=(state.detach().float()-self.state_mean)/self.state_scale/np.sqrt(self.state_dim)*self.state_weight
        return self.head(torch.cat([a,b],dim=-1))

def flight_weights(groups):
    groups=np.asarray(groups);unique=np.unique(groups)
    return np.array([1/(len(unique)*np.sum(groups==g)) for g in groups])

def fit_state_critic(features,state,targets,groups,alpha,state_weight):
    x=np.asarray(features,dtype=float);z=np.asarray(state,dtype=float);y=np.asarray(targets,dtype=float);g=np.asarray(groups)
    if x.ndim!=2 or z.ndim!=2 or y.shape!=(len(x),) or len(z)!=len(x) or len(g)!=len(x) or not len(x):
        raise ValueError('invalid critic data shape')
    if not all(np.isfinite(a).all() for a in [x,z,y]) or alpha<=0 or state_weight<=0:
        raise ValueError('finite data and positive regularization required')
    w=flight_weights(g);mean=w@z;std=np.sqrt(w@((z-mean)**2));std=np.where(std<1e-6,1.,std)
    design=np.concatenate([x/np.sqrt(x.shape[1]),(z-mean)/std/np.sqrt(z.shape[1])*state_weight],axis=1)
    mx=w@design;my=w@y;center=(design-mx)*np.sqrt(w[:,None]);rhs=(y-my)*np.sqrt(w)
    weight=center.T@np.linalg.solve(center@center.T+alpha*np.eye(len(x)),rhs);bias=my-mx@weight
    model=StateCritic(x.shape[1],z.shape[1])
    with torch.no_grad():
        model.state_mean.copy_(torch.from_numpy(mean));model.state_scale.copy_(torch.from_numpy(std));model.state_weight.fill_(state_weight)
        model.head.weight.copy_(torch.from_numpy(weight[None,:]));model.head.bias.fill_(float(bias))
    return model

def select_state_critic(features,state,targets,groups,alphas,state_weights):
    x=np.asarray(features);z=np.asarray(state);y=np.asarray(targets);g=np.asarray(groups);rows=[]
    if len(np.unique(g))<3:raise ValueError('whole-flight CV requires three flights')
    for sw in state_weights:
        for alpha in alphas:
            folds=[]
            for held in np.unique(g):
                train=g!=held;val=~train
                h=fit_state_critic(x[train],z[train],y[train],g[train],alpha,sw)
                with torch.no_grad():pred=h(torch.as_tensor(x[val],dtype=torch.float32),torch.as_tensor(z[val],dtype=torch.float32)).flatten().numpy()
                folds.append(dict(heldout_group=str(held),training_groups=[str(a) for a in np.unique(g[train])],mse=float(np.mean((pred-y[val])**2))))
            rows.append(dict(alpha=float(alpha),state_weight=float(sw),mse=float(np.mean([f['mse'] for f in folds])),folds=folds))
    best=min(rows,key=lambda a:(a['mse'],-a['alpha'],abs(np.log(a['state_weight']))))
    return fit_state_critic(x,z,y,g,best['alpha'],best['state_weight']),dict(selected_alpha=best['alpha'],selected_state_weight=best['state_weight'],validation_mse=best['mse'],candidates=rows)
