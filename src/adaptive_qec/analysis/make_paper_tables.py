"""Generate manuscript tables from the corrected study's integer error counts and host traces."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from adaptive_qec.analysis.causal_results import summarize

NAMES={"static":"Factory MWPM","raw":"Raw update","shrink":"50:50 shrinkage","rolling":"Two-window shrinkage",
       "periodic":"Periodic, capped","selective":"Selective, capped","legacy":"Historical gate, causal",
       "matched":"Matched-model reference","publisher_correlated":"Published correlated matching"}


def table(headers, rows, caption):
    return "\n".join([r"\begin{center}\small", r"\begin{tabular}{l"+"r"*(len(headers)-1)+"}",
                      r"\hline"," & ".join(headers)+r" \\",r"\hline",
                      *[" & ".join(map(str,r))+r" \\" for r in rows],r"\hline\end{tabular}",
                      r"\par\smallskip\parbox{0.93\linewidth}{\small "+caption+"}",r"\end{center}",""])


def generate(results, config, public, timing, output):
    s=summarize(json.loads(results.read_text()),yaml.safe_load(config.read_text()))
    output.mkdir(parents=True,exist_ok=True)
    def write(name,value):
        (output/name).write_text(value,encoding="utf8",newline="\n")
    def panel(w,sc):
        return next(p for p in s["panels"] if (p["distance"],p["basis"],p["window"],p["scenario"])==(3,"Z",w,sc))
    w=panel(2000,"hotspot")
    rows=[[NAMES[a],panel(200,"hotspot")["errors"][a],v,f'{100*w["ler"][a]:.3f}',w["fits"].get(a,"--")]
          for a,v in w["errors"].items()]
    write("table_1.tex",table(["Decoder","Errors, 200","Errors, 2000",r"LER (\%), 2000","Fits, 2000"],rows,
          f"Distance-three Z memory, transient hotspot. Each column scores the same {w['shots']:,} shots. Window sizes are in shots."))
    rows=[]
    for sc,padj in s["holm"].items():
        p=panel(2000,sc)
        c=p["selective_minus"]["periodic"]
        rows.append([sc,f'{100*c["mean"]:+.3f}',f'[{100*c["ci95"][0]:+.3f}, {100*c["ci95"][1]:+.3f}]',
                     f'{c["p"]:.3f}',f'{padj:.3f}'])
    write("table_2.tex",table(["Regime",r"$\Delta$ (pp)","95\\% CI (pp)","Raw $p$","Holm $p$"],rows,
          "Selective minus periodic LER. Points and intervals are paired across 30 trajectories. Holm covers the four rows."))
    records=json.loads(public.read_text())["records"]
    rows=[[NAMES[a],*[sum(r["errors"][a]) for r in records]] for a in records[0]["errors"]]
    write("table_3.tex",table(["Decoder",*[f'{r["basis"]}, {r["window"]}' for r in records]],rows,
          f"Google memory data. Errors per {records[0]['shots']:,} scored shots per basis. Columns with the same basis reuse the same shots."))
    ts=json.loads(timing.read_text())
    rows=[]
    for r in ts["records"]:
        if (r["distance"],r["window"])!=(3,2000):
            continue
        for act in sorted({t["action"] for t in r["samples"]}):
            vals=[v for v in r["samples"] if v["action"]==act]
            rows.append([act,len(vals),*[f'{np.median([v[k] for v in vals]):.0f}' for k in
                        ("decode_us","monitor_us","update_us","total_us")],
                         f'{np.percentile([v["total_us"] for v in vals],95):.0f}'])
    write("table_4.tex",table(["Action","Trials","Decode","Monitor","Update","Total","Total p95"],rows,
          r"Host microseconds per complete window, $d=3$, $W=2000$. Stage and total medians are computed separately."))
    rows=[]
    for p in s["panels"]:
        if p["window"]==2000 and p["scenario"]=="hotspot":
            c=p["selective_minus"]["periodic"]
            rows.append([p["distance"],p["basis"],p["seeds"],p["errors"]["selective"],p["errors"]["periodic"],
                         f'{100*c["mean"]:+.3f}'])
    write("table_5.tex",table(["Distance","Basis","Seeds","Selective","Periodic",r"$\Delta$ (pp)"],rows,
          "Hotspot transfer panels at 2000 shots per window. Secondary panels are descriptive."))
    return s


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results",type=Path,default=Path("data/suite_results.json"))
    ap.add_argument("--output-dir",type=Path,default=Path("papers/tables"))
    a=ap.parse_args()
    generate(a.results,Path("configs/suite.yaml"),Path("data/public_replay_results.json"),
             Path("data/host_timings.json"),a.output_dir)


if __name__=="__main__":
    main()
