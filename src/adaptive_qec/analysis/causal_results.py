"""Compute all manuscript endpoints from per-window integer error and action records."""
from collections import defaultdict

import numpy as np
from scipy import stats

from adaptive_qec.experiments.causal_reweighting import ARMS


def signed_rank_p(values):
    """Conditional two-sided sign randomization of the nonzero signed ranks, including ties."""
    differences = np.round(values, 12)
    differences = differences[differences != 0]
    if len(differences) == 0:
        return 1.0
    ranks = (2 * stats.rankdata(abs(differences))).astype(int)
    observed = int(ranks[differences > 0].sum())
    total = int(ranks.sum())
    ways = [0] * (total + 1)
    ways[0] = 1
    for rank in ranks:
        for score in range(total, rank - 1, -1):
            ways[score] += ways[score-rank]
    tail = sum(ways[:min(observed, total-observed)+1])
    return min(1.0, 2 * tail / 2**len(ranks))


def contrast(values):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    se = float(stats.sem(x))
    crit = stats.t.ppf(0.975,len(x)-1)
    if se:
        p = float(stats.ttest_1samp(x,0).pvalue)
    else:
        p = 1.0 if mean == 0 else 0.0
    return dict(mean=mean, ci95=[mean-crit*se,mean+crit*se],
                upper95=mean+stats.t.ppf(0.95,len(x)-1)*se,p=p,
                signed_rank_p=signed_rank_p(x),
                wins=int(sum(x<0)), ties=int(sum(x==0)), losses=int(sum(x>0)))


def holm(pvals):
    order=sorted(pvals,key=pvals.get)
    adj={}
    highest=0.0
    for i,k in enumerate(order):
        highest=max(highest,min(1.0,(len(order)-i)*pvals[k]))
        adj[k]=highest
    return adj


def summarize(data, cfg):
    groups=defaultdict(list)
    seen=set()
    for row in data["trajectories"]:
        key=(row["distance"],row["basis"],row["window"],row["scenario"])
        identity=(*key,row["seed"])
        if identity in seen:
            raise ValueError(f"Duplicate trajectory {identity}")
        seen.add(identity)
        groups[key].append(row)
    expected={(p["distance"],p["basis"],w,s,seed) for p in cfg["panels"] for w in cfg["windows"]
              for s in cfg["regimes"] for seed in range(p["first_seed"],p["first_seed"]+p["count"])}
    if seen!=expected:
        raise ValueError("Incomplete evaluation grid")
    panels=[]
    primary=cfg["primary"]
    family={}
    for (d,b,w,s),rows in sorted(groups.items()):
        ler={arm:[] for arm in ARMS}
        counts={arm:0 for arm in ARMS}
        updates={arm:0 for arm in ARMS if arm!="matched"}
        resets=updates.copy()
        for r in rows:
            if r["shots"] != cfg["block_shots"] * cfg["blocks"] or r["shots"] % w:
                raise ValueError("Invalid shot denominator")
            if set(r["errors"]) != set(ARMS):
                raise ValueError("Incorrect decoder set")
            windows=r["shots"]//w
            if len(r["actions"])!=windows or any(len(v)!=windows for v in r["errors"].values()):
                raise ValueError("Window count mismatch")
            for arm in ARMS:
                v=r["errors"][arm]
                if any(type(e) is not int or not 0<=e<=w for e in v):
                    raise ValueError("Invalid logical error count")
                if arm not in ("matched","static") and v[0]!=r["errors"]["static"][0]:
                    raise ValueError("First window did not use the factory graph")
                counts[arm]+=sum(v)
                ler[arm].append(sum(v)/r["shots"])
                if arm!="matched":
                    actions=[a[arm] for a in r["actions"]]
                    if not set(actions) <= {"stay", "fit", "reset"} or (arm == "static" and set(actions) != {"stay"}):
                        raise ValueError("Unexpected decoder action")
                    uses=actions.count("fit")
                    if actions[-1]!="stay" or arm in ("selective","periodic") and uses>cfg["budget"]:
                        raise ValueError("Invalid update schedule")
                    updates[arm]+=uses
                    resets[arm]+=actions.count("reset")
        diffs={arm:contrast(np.array(ler["selective"])-ler[arm]) for arm in ("periodic","static","shrink","rolling")}
        panel=dict(distance=d,basis=b,window=w,scenario=s,seeds=len(rows),shots=sum(r["shots"] for r in rows),
                   errors=counts,ler={k:float(np.mean(v)) for k,v in ler.items()},fits=updates,
                   resets=resets,selective_minus=diffs)
        panels.append(panel)
        if (d,b,w)==(primary["distance"],primary["basis"],primary["window"]):
            family[s]=panel
    adjusted=holm({s:v["selective_minus"][primary["comparator"]]["p"] for s,v in family.items()})
    target=family[primary["scenario"]]["selective_minus"][primary["comparator"]]
    safe=family["stationary"]["selective_minus"]["static"]
    success=(adjusted[primary["scenario"]]<cfg["alpha"] and target["mean"]<=-cfg["safety_margin"]
             and safe["upper95"]<cfg["safety_margin"])
    return dict(panels=panels,primary=target,control=safe,holm=adjusted,primary_success=bool(success))
