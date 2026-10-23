"""Direct high precision truth, precision ladder, deterministic pixel populations."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import mpmath as mp
import numpy as np
from . import hp

SAMPLE = [(round(i*1919/24),round(j*1079/24)) for i in range(25) for j in range(25)]


def _verdict(re_s,im_s,max_iter,digits,shortcut):
    ctx = mp.mp.clone()
    ctx.dps = digits
    c = ctx.mpc(re_s,im_s)
    if not c:
        return False,max_iter
    z, lc = c,ctx.log(c)
    ones = int(z==1)
    for n in range(1,max_iter+1):
        z = ctx.exp(z*lc)
        if abs(z)>ctx.mpf('1e10'):
            return True,n
        if z==1:
            ones += 1
            if shortcut and ones==2:
                # Identical values at fixed precision have identical successors.
                # No escape occurred between the two ones. No tolerance is used.
                return False,max_iter
    return False,max_iter


def verdict(re_s,im_s,max_iter,digits):
    """Escape power n>=1; otherwise stop=N. Only a second exact 1 shortcuts."""
    return _verdict(re_s,im_s,max_iter,digits,True)


def pixel_points(x0,y0,eps,pixels):
    return [(str(hp.grid_x(x0,eps,int(p))),str(hp.grid_y(y0,eps,int(q)))) for p,q in pixels]


def sample_points(x0,y0,eps):
    return pixel_points(x0,y0,eps,SAMPLE)


def seed_for(name,eps):
    return int.from_bytes(hashlib.sha256((str(name)+'|'+float(eps).hex()).encode()).digest()[:8],'little')


def uniform_pixels(seed,n):
    if not 0<=n<=1920*1080:
        raise ValueError('uniform sample size outside population')
    flat = np.random.default_rng(seed).choice(1920*1080,size=n,replace=False)
    return [(int(v//1080),int(v%1080)) for v in flat]


def band_mask(bool_map):
    """Union of -3..+3 about the lower endpoint of every transition pair."""
    m = np.asarray(bool_map,dtype=bool)
    if m.ndim!=2:
        raise ValueError('map must have two axes')
    mask = np.zeros(m.shape,dtype=bool)
    for axis in (0,1):
        a,b = np.nonzero(np.diff(m,axis=axis))
        for d in range(-3,4):
            x,y = (a+d,b) if axis==0 else (a,b+d)
            ok = (x>=0)&(x<m.shape[0])&(y>=0)&(y<m.shape[1])
            mask[x[ok],y[ok]] = True
    return mask


def band_pixels(bool_map,seed,n):
    flat = np.flatnonzero(band_mask(bool_map))
    chosen = np.random.default_rng(seed).choice(flat,size=min(n,len(flat)),replace=False)
    ny = np.shape(bool_map)[1]
    return [(int(v//ny),int(v%ny)) for v in chosen]


def _worker(args):
    return verdict(*args)


def _batch(points,max_iter,digits,processes,cache_dir,pool=None):
    """Per-coordinate cache: independent of batch membership or sample order."""
    values, missing, paths = [None]*len(points), [], {}
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True,exist_ok=True)
    for i,(x,y) in enumerate(points):
        key = [str(x),str(y),int(max_iter),int(digits),'exp-second-one-v2']
        path = None
        if cache_dir is not None:
            digest = hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).hexdigest()
            path = Path(cache_dir)/digest[:2]/(digest+'.json')
            if path.exists():
                data = json.loads(path.read_text(encoding='utf-8'))
                if data['key']==key:
                    values[i] = tuple(data['value'])
                    continue
        missing.append((i,(str(x),str(y),max_iter,digits)))
        paths[i] = (path,key)
    args = [a for _,a in missing]
    if pool is not None:
        computed = pool.map(_worker,args,chunksize=8)
    elif processes==1 or not args:
        computed = map(_worker,args)
    else:
        with ProcessPoolExecutor(max_workers=processes,mp_context=multiprocessing.get_context('spawn')) as workers:
            computed = list(workers.map(_worker,args,chunksize=8))
    for (i,_),value in zip(missing,computed):
        values[i] = value
        path,key = paths[i]
        if path is not None:
            path.parent.mkdir(parents=True,exist_ok=True)
            # Atomic publication: interrupted workers cannot leave partial JSON.
            import os
            tmp = path.with_suffix('.'+str(os.getpid())+'.tmp')
            tmp.write_text(json.dumps({'key':key,'value':value}),encoding='utf-8')
            tmp.replace(path)
    return values,len(missing)


def ladder(points,max_iter,depth_digits,processes=12,cache_dir=None):
    """Return JSON-ready per-point decisions and measured throughput diagnostics."""
    if processes<1:
        raise ValueError('processes must be positive')
    points = [(str(x),str(y)) for x,y in points]
    d1 = int(depth_digits)+20
    d2,d3 = d1+30,d1+90
    start = time.perf_counter()
    pool = (ProcessPoolExecutor(max_workers=processes,mp_context=multiprocessing.get_context('spawn'))
            if processes>1 and points else None)
    try:
        a,na = _batch(points,max_iter,d1,processes,cache_dir,pool)
        b,nb = _batch(points,max_iter,d2,processes,cache_dir,pool)
        different = [i for i in range(len(points)) if a[i]!=b[i]]
        c,nc = _batch([points[i] for i in different],max_iter,d3,processes,cache_dir,pool)
    finally:
        if pool is not None:
            pool.shutdown()
    thirds = dict(zip(different,c))
    rows = []
    for i,(x,y) in enumerate(points):
        value, digits, unstable = b[i],d2,False
        if i in thirds:
            value, digits = thirds[i],d3
            unstable = b[i]!=value
        rows.append({'decided':not unstable,'diverged':bool(value[0]) if not unstable else None,
                     'stop':int(value[1]) if not unstable else None,'digits':digits,'unstable':unstable,
                     'd1':list(a[i]),'d2':list(b[i]),'d3':list(thirds[i]) if i in thirds else None})
    seconds = time.perf_counter()-start
    return {'points':rows,'digits':[d1,d2,d3],'seconds':seconds,
            'points_per_second':len(points)/seconds if seconds else 0.,
            'computed_verdicts':na+nb+nc,'cached_verdicts':2*len(points)+len(different)-na-nb-nc,
            'd3_points':len(different)}


def truth_samples(x0,y0,eps,max_iter,digits,processes=14,cache_dir=None):
    values,_ = _batch(sample_points(x0,y0,eps),max_iter,digits,processes,cache_dir)
    return np.asarray([v[0] for v in values],dtype=bool), np.asarray([v[1] for v in values],dtype=np.int32)
