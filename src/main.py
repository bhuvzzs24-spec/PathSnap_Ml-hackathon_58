from pathlib import Path
import io, json, os, time, zipfile
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics import f1_score
from features import FEATURES, build_features, reference_labels
from hmm import decode, continuity

ROOT=Path(__file__).resolve().parent
PARTITION=os.getenv('PATHSNAP_PARTITION','student_test')
DATA=Path(os.getenv('PATHSNAP_DATA',str(ROOT/'Student folder.zip')))
ARTIFACT=ROOT/'models'
app=FastAPI(title='PathSnap',version='1.0')
model=None; config=None

def load_model():
    global model,config
    if model is None:
        cfg=json.loads((ARTIFACT/'config.json').read_text())
        cols=cfg.get('feature_indices',list(range(len(FEATURES))))
        if cfg['K']!=12 or [FEATURES[j] for j in cols]!=cfg['features']:
            raise ValueError('Model config does not match the original feature pipeline')
        try:
            from xgboost import XGBClassifier
            m=XGBClassifier(); m.load_model(str(ARTIFACT/'xgboost_model.json'))
        except ImportError:
            from json_predictor import JSONPredictor
            m=JSONPredictor(ARTIFACT/'xgboost_model.json')
        if m.n_features_in_!=len(cols): raise ValueError('Model feature count mismatch')
        config=cfg; model=m
    return model,config

def inventory():
    with zipfile.ZipFile(DATA) as z:
        return {Path(n).stem:n.rsplit('.',1)[0] for n in z.namelist() if n.endswith('.track') and PARTITION in Path(n).parts}

def load_trip(tid):
    base=inventory()[tid]
    with zipfile.ZipFile(DATA) as z:
        def read(suffix,dtype=float): return np.loadtxt(io.BytesIO(z.read(base+suffix)),dtype=dtype,ndmin=2)
        track=read('.track'); nodes=read('.nodes'); arcs=read('.arcs',int)
    track=track[np.isfinite(track).all(axis=1)]; track=track[np.argsort(track[:,2],kind='stable')]
    if not len(track): raise ValueError('Empty GPS track')
    track=track[np.r_[True,np.diff(track[:,2])>0]]
    if len(track)==0 or arcs.min()<0 or arcs.max()>=len(nodes): raise ValueError('Invalid trip')
    return track,nodes,arcs,base

def predict(tid, penalty=2.0):
    start=time.perf_counter(); m,cfg=load_model(); track,nodes,arcs,base=load_trip(tid)
    X,C,I,P,A,B=build_features(track,nodes,arcs,1)
    cols=cfg.get('feature_indices',list(range(len(FEATURES))))
    prob=m.predict_proba(X.reshape(-1,len(FEATURES))[:,cols])[:,1].reshape(C.shape)
    raw=C[np.arange(len(C)),prob.argmax(axis=1)]
    decoded=decode(C,prob,arcs,track[:,2],penalty)
    elapsed=time.perf_counter()-start
    metrics={'top1_accuracy':None,'macro_f1':None,'route_continuity':continuity(decoded,arcs),'inference_seconds':elapsed,'classifier_top1_accuracy':None,'classifier_route_continuity':continuity(raw,arcs),'candidate_miss_rate':None}
    # Reference is read only AFTER prediction. Never return labels or route geometry.
    with zipfile.ZipFile(DATA) as z:
        if base+'.route' in z.namelist():
            route=np.loadtxt(io.BytesIO(z.read(base+'.route')),dtype=int,ndmin=1)
            ref=reference_labels(P,A,B,route,I)
            metrics.update(top1_accuracy=float(np.mean(decoded==ref)),classifier_top1_accuracy=float(np.mean(raw==ref)),candidate_miss_rate=float(np.mean(~(C==ref[:,None]).any(1))),macro_f1=float(f1_score((C==ref[:,None]).ravel(),(C==decoded[:,None]).ravel(),average='macro',zero_division=0)))
    def segments(ids):
        unique=ids[np.r_[True,ids[1:]!=ids[:-1]]]
        return nodes[arcs[unique]][:,:,::-1].tolist()
    return {'trip_id':tid,'model':'XGBoost + HMM/Viterbi','gps_points':track[:,[1,0]].tolist(),'predicted_route':segments(decoded),'classifier_route':segments(raw),'metrics':metrics,'partition':PARTITION}

class Request(BaseModel):
    trip_id:str

@app.get('/health')
def health():
    try:
        load_model(); trips=inventory()
        return {'status':'ok','partition':PARTITION,'trip_count':len(trips)}
    except Exception as e: raise HTTPException(503,detail=str(e))

@app.get('/trips')
def trips():
    try: return {'trip_ids':sorted(inventory())}
    except Exception as e: raise HTTPException(503,detail=str(e))

@app.post('/predict')
def run(request:Request):
    try:
        if request.trip_id not in inventory(): raise HTTPException(404,'Unknown trip ID')
        return predict(request.trip_id,float(os.getenv('PATHSNAP_PENALTY','2.0')))
    except HTTPException: raise
    except Exception as e: raise HTTPException(500,detail=str(e))
