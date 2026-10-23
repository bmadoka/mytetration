"""Production hybrid rules. No CUDA or GL imports; all coordinates are host float64."""
import statistics

import numpy as np


def decide_mode(previous, finished):
    """Mode decision (fresh-only / warped; replayed as condition4 by tests/viewer_check.py), called immediately before
    presenting with available finished records."""
    durations = [r["t_end"] - r["t_start"] for r in finished[-8:]]
    m = statistics.median(durations) if durations else None
    mode = previous
    if m is None:
        mode = "fresh-only"
    elif previous == "fresh-only" and m > 0.033:  # HYBRID_ENTER
        mode = "warped"
    elif previous == "warped" and m <= 0.028:
        mode = "fresh-only"
    return mode, m


def warp_indices(display, source, transform=None, *, clamp=True, margin=(0, 0)):
    """Hybrid display rule: integer source axes, scale and offsets in source pixels."""
    xd, yd, ed = map(float, display)
    xs, ys, es = map(float, source)
    eyd, eys = ed * (9 / 16), es * (9 / 16)
    ox = ((xd - xs) - ed + es) / (2 * es / 1919)
    oy = ((yd - ys) - eyd + eys) / (2 * eys / 1079)
    sx = ((xd - xs) - ed + es + np.arange(1920, dtype=np.float64) * (2 * ed / 1919)) / (2 * es / 1919)
    sy = ((yd - ys) - eyd + eys + np.arange(1080, dtype=np.float64) * (2 * eyd / 1079)) / (2 * eys / 1079)
    if transform is not None:
        # Apply recorded affine values relative to the canonical axes. Equal values
        # add exactly zero: the prescribed expression order and half ties survive.
        scale, offset = transform
        sx += (float(offset[0]) - ox) + np.arange(1920) * (float(scale) - ed / es)
        sy += (float(offset[1]) - oy) + np.arange(1080) * (float(scale) - ed / es)
    mx, my = margin
    ix = np.clip(np.floor(sx + 0.5), -mx if clamp else -mx-1, 1919+mx if clamp else 1920+mx).astype(np.int32)
    iy = np.clip(np.floor(sy + 0.5), -my if clamp else -my-1, 1079+my if clamp else 1080+my).astype(np.int32)
    return ix, iy, ed / es, (ox, oy)


def same_params(a, b):
    return tuple(float(v).hex() for v in a) == tuple(float(v).hex() for v in b)


def duplicate_params(p, computing, finished):
    return (any(same_params(p, r["params"]) for r in computing)
            or bool(finished and same_params(p, finished[-1]["params"])))


def coverage_counts(axes):
    """Exact priority rectangle areas, without constructing a render-target mask."""
    rectangles = []
    for ix, iy in axes:
        x, y = np.flatnonzero((ix >= 0) & (ix < 1920)), np.flatnonzero((iy >= 0) & (iy < 1080))
        rectangles.append((int(x[0]), int(x[-1])+1, int(y[0]), int(y[-1])+1) if x.size and y.size else (0, 0, 0, 0))
    xs = sorted({0, 1920, *(v for r in rectangles for v in r[:2])})
    ys = sorted({0, 1080, *(v for r in rectangles for v in r[2:])})
    occupied = np.zeros((len(xs)-1, len(ys)-1), bool)
    areas = np.diff(xs)[:, None] * np.diff(ys)[None, :]
    counts = []
    for x0, x1, y0, y1 in rectangles:
        mask = ((np.asarray(xs[:-1])[:, None] >= x0) & (np.asarray(xs[1:])[:, None] <= x1)
                & (np.asarray(ys[:-1])[None, :] >= y0) & (np.asarray(ys[1:])[None, :] <= y1))
        counts.append(int(areas[mask & ~occupied].sum()))
        occupied |= mask
    return {"newest": counts[0], "older": sum(counts[1:]), "edge": int(areas[~occupied].sum())}


def candidate_plan(display, sources, transforms=None):
    """Newest-first source records -> integer axes, logged transforms and exact counts."""
    axes, records = [], []
    for i, src in enumerate(sources):
        tr = None if transforms is None else (transforms[i]["scale"], transforms[i]["offset"])
        ix, iy, scale, offset = warp_indices(display, src["params"], tr, clamp=False)
        axes.append((ix, iy))
        records.append({"id": src["id"], "scale": scale, "offset": list(offset)})
    return axes, records, coverage_counts(axes)


def fresh_candidate_records(display, sources):
    """candidate_plan records/counts for a fresh present without index arrays (0.39 ms -> scalars).
    Only when the newest source has the display parameters (bitwise): identity transform, so the newest covers every
    pixel. Scale/offset use the expressions of warp_indices (same bits). Otherwise None (caller uses candidate_plan)."""
    if not sources or not same_params(display, sources[0]["params"]):
        return None
    xd, yd, ed = map(float, display)
    records = []
    for src in sources:
        xs, ys, es = map(float, src["params"])
        eyd, eys = ed * (9 / 16), es * (9 / 16)
        records.append({"id": src["id"], "scale": ed / es,
                        "offset": [((xd - xs) - ed + es) / (2 * es / 1919), ((yd - ys) - eyd + eys) / (2 * eys / 1079)]})
    return records, {"newest": 1920 * 1080, "older": 0, "edge": 0}


def motion_candidates(shown, precise, t_decide, margin_map):
    """B3 ordered (descriptor, immutable map) pairs, from actual first-show history."""
    result=[]
    for _, rec, screen in sorted(shown,key=lambda item:(item[0],item[1]["id"]),reverse=True)[:64]:
        usable = rec.get("margin_t_end") is not None and rec["margin_t_end"] < t_decide  # MARGIN_USABLE
        part = "screen+margin" if usable else "screen"
        descriptor = dict(rec, part=part, margin=rec.get("margin_width",[0,0]) if usable else [0,0])
        result.append((descriptor,margin_map(rec["id"]) if usable else screen))
    if precise is not None:
        rec,m=precise
        result.append((dict(rec,part="precise",margin=[0,0]),m))
    return result


def candidate_plan3(display, sources, transforms=None):
    axes,records,rects=[],[],[]
    def rectangle(ix,iy,bounds):
        x0,x1,y0,y1=bounds
        x=np.flatnonzero((ix>=x0)&(ix<x1)); y=np.flatnonzero((iy>=y0)&(iy<y1))
        return (int(x[0]),int(x[-1])+1,int(y[0]),int(y[-1])+1) if x.size and y.size else (0,0,0,0)
    for i,src in enumerate(sources):
        mx,my=src.get("margin",(0,0))
        tr=None if transforms is None else (transforms[i]["scale"],transforms[i]["offset"])
        ix,iy,scale,offset=warp_indices(display,src["params"],tr,clamp=False,margin=(mx,my))
        axes.append((ix,iy))
        records.append({"id":src["id"],"scale":scale,"offset":list(offset)})
        rects.append((rectangle(ix,iy,(-mx,1920+mx,-my,1080+my)),rectangle(ix,iy,(0,1920,0,1080))))
    # Subdivide at every rectangle boundary; exact areas, no 2MP per-frame mask.
    xs=sorted({0,1920,*(v for pair in rects for r in pair for v in r[:2])})
    ys=sorted({0,1080,*(v for pair in rects for r in pair for v in r[2:])})
    xx,yy=np.asarray(xs),np.asarray(ys)
    areas=np.diff(xx)[:,None]*np.diff(yy)[None,:]
    remaining=np.ones(areas.shape,bool)
    counts=dict(newest=0,older=0,margin=0,precise=0,edge=0)
    def mask(r):
        a,b,c,d=r
        return (xx[:-1,None]>=a)&(xx[1:,None]<=b)&(yy[None,:-1]>=c)&(yy[None,1:]<=d)
    for i,(src,(extended,screen)) in enumerate(zip(sources,rects)):
        take=remaining & mask(extended)
        if src["part"]=="precise":
            counts["precise"]+=int(areas[take].sum())
        else:
            inside=mask(screen)
            counts["newest" if i==0 else "older"]+=int(areas[take & inside].sum())
            counts["margin"]+=int(areas[take & ~inside].sum())
        remaining &= ~take
    counts["edge"]=int(areas[remaining].sum())
    return axes,records,counts
