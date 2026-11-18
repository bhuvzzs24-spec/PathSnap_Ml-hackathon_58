import numpy as np
from scipy.spatial import cKDTree
K=12
N_JOBS=4

def project(lonlat, origin):
    scale=np.array([111195*np.cos(np.deg2rad(origin[1])),111195])
    return (lonlat-origin)*scale

def geometry(p,a,b):
    v=b-a
    length=np.linalg.norm(v,axis=-1)
    u=np.clip(np.sum((p-a)*v,axis=-1)/np.maximum(length**2,1e-9),0,1)
    q=a+u[...,None]*v
    return np.linalg.norm(p-q,axis=-1),u,length,q

def index_segments(a,b):
    count=np.maximum(2,np.ceil(np.linalg.norm(b-a,axis=1)/20).astype(int)+1)
    ids=np.repeat(np.arange(len(a)),count)
    offsets=np.cumsum(count)-count
    frac=(np.arange(len(ids))-np.repeat(offsets,count))/np.repeat(count-1,count)
    samples=a[ids]+frac[:,None]*(b[ids]-a[ids])
    return cKDTree(samples),ids

def candidates(points,a,b,k):
    k=min(k,len(a)); tree,ids=index_segments(a,b)
    query_n=min(96,len(ids))
    result=np.empty((len(points),k),dtype=np.int32)
    for start in range(0,len(points),1024):
        chunk=points[start:start+1024]
        _,idx=tree.query(chunk,k=query_n,workers=N_JOBS)
        idx=np.asarray(idx).reshape(len(chunk),query_n)
        for j,p in enumerate(chunk):
            pool=np.unique(ids[idx[j]])
            if len(pool)<k: pool=np.arange(len(a))
            d,*_=geometry(p,a[pool],b[pool])
            order=np.argsort(d,kind='stable')[:k]
            result[start+j]=pool[order]
    return result

FEATURES=['distance_m','distance_rank','distance_gap_m','segment_length_m','projection_fraction','heading_cos','speed_mps','time_gap_s','prev_distance_m','next_distance_m','prev_heading_cos','next_heading_cos','endpoint_distance_m','step_m','local_turn_cos']

def unit(v):
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9)

def build_features(track,nodes,arcs,stride=1):
    origin=np.median(track[:,:2],axis=0)
    points=project(track[:,:2],origin); xy=project(nodes,origin)
    a,b=xy[arcs[:,0]],xy[arcs[:,1]]
    prev=np.vstack([points[0],points[:-1]]); nxt=np.vstack([points[1:],points[-1]])
    vin=points-prev; vout=nxt-points
    step=np.linalg.norm(vin,axis=1); dt=np.r_[1,np.diff(track[:,2])]
    speed=step/np.maximum(dt,1e-3)
    take=np.arange(0,len(points),stride)
    p=points[take]; cand=candidates(p,a,b,K)
    aa,bb=a[cand],b[cand]; d,u,length,q=geometry(p[:,None,:],aa,bb)
    direction=unit(bb-aa)
    def cos(v): return np.sum(unit(v[take])[:,None,:]*direction,axis=-1)
    prev_d=geometry(prev[take,None,:],aa,bb)[0]
    next_d=geometry(nxt[take,None,:],aa,bb)[0]
    def broad(v): return np.broadcast_to(v[take,None],d.shape)
    matrix=np.stack([d,np.broadcast_to(np.arange(cand.shape[1]),d.shape),d-d[:,:1],length,u,cos(vin+vout),broad(speed),broad(dt),prev_d,next_d,cos(vin),cos(vout),np.minimum(u,1-u)*length,broad(step),broad(np.sum(unit(vin)*unit(vout),axis=-1))],axis=-1).astype(np.float32)
    return matrix,cand,take,points,a,b

def reference_labels(points,a,b,route,take):
    # Restrict label assignment to supplied reference arcs; independent of candidate generation.
    # Nearest reference arc with a modest direction penalty; stationary points use distance alone.
    ids=np.unique(route)
    prev=np.vstack([points[0],points[:-1]])
    nxt=np.vstack([points[1:],points[-1]])
    p=points[take]; nearby=candidates(p,a[ids],b[ids],min(12,len(ids)))
    r=ids[nearby]; aa,bb=a[r],b[r]
    d=geometry(p[:,None,:],aa,bb)[0]
    v=(nxt-prev)[take]; heading=np.sum(unit(v)[:,None,:]*unit(bb-aa),axis=-1)
    penalty=np.where(np.linalg.norm(v,axis=1)[:,None]>3,3*(1-heading),0)
    return r[np.arange(len(p)),np.argmin(d+penalty,axis=1)]