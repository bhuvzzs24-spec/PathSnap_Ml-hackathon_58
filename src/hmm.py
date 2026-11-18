"""Log-space Viterbi with locally normalized transition scores."""
import numpy as np

def continuity(seq,arcs):
    if len(seq)<2:return 1.0
    return float(np.mean((seq[:-1]==seq[1:])|(arcs[seq[:-1],1]==arcs[seq[1:],0])))

def decode(candidates, probabilities, arcs, timestamps, penalty=2.0):
    candidates=np.asarray(candidates); n,k=candidates.shape
    if n==0: return np.empty(0,dtype=int)
    # Candidate scores normalized within observation; zero probabilities clipped.
    p=np.clip(probabilities,1e-9,1); emission=np.log(p/p.sum(axis=1,keepdims=True))
    score=emission[0].copy(); back=np.zeros((n,k),dtype=np.int32)
    for t in range(1,n):
        old,new=candidates[t-1],candidates[t]
        same=old[:,None]==new[None,:]
        adjacent=arcs[old,1,None]==arcs[new,0][None,:]
        # Larger gaps relax connectivity; disconnected transitions remain possible.
        gap=max(float(timestamps[t]-timestamps[t-1]),1.0)
        cost=penalty/min(max(gap/5,1),5)
        trans=np.where(same|adjacent,0.0,-cost)
        trans-=np.log(np.exp(trans).sum(axis=1,keepdims=True))
        v=score[:,None]+trans
        back[t]=np.argmax(v,axis=0)
        score=emission[t]+v[back[t],np.arange(k)]
        score-=score.max()
    idx=np.empty(n,dtype=int); idx[-1]=score.argmax()
    for t in range(n-1,0,-1): idx[t-1]=back[t,idx[t]]
    return candidates[np.arange(n),idx]
