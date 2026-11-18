"""Offline inference for numeric, binary-logistic XGBoost JSON tree ensembles."""
import json
import numpy as np

class JSONPredictor:
    def __init__(self,path):
        self.raw=json.loads(path.read_text())['learner']
        p=self.raw['learner_model_param']; b=self.raw['gradient_booster']
        if self.raw['objective']['name']!='binary:logistic' or b['name']!='gbtree':
            raise ValueError('Unsupported objective or booster')
        base=float(json.loads(p['base_score'])[0] if p['base_score'].startswith('[') else p['base_score'])
        self.margin=np.float32(np.log(base/(1-base))); self.n_features_in_=int(p['num_feature']); self.trees=[]
        for t in b['model']['trees']:
            if any(t['split_type']) or t['categories']: raise ValueError('Categorical splits unsupported')
            self.trees.append(tuple(np.asarray(t[k],dtype=dt) for k,dt in [('left_children',int),('right_children',int),('split_indices',int),('split_conditions',np.float32),('default_left',bool)]))
    def predict_proba(self,X):
        X=np.asarray(X,dtype=np.float32); total=np.full(len(X),self.margin,dtype=np.float32); rows=np.arange(len(X))
        for left,right,feat,split,default in self.trees:
            node=np.zeros(len(X),dtype=np.int32)
            while True:
                active=left[node]!=-1
                if not active.any(): break
                r=rows[active]; n=node[active]; x=X[r,feat[n]]
                go=np.where(np.isnan(x),default[n],x<split[n]); node[active]=np.where(go,left[n],right[n])
            total+=split[node]
        prob=1/(1+np.exp(-np.clip(total,-80,80)))
        return np.column_stack((1-prob,prob))
