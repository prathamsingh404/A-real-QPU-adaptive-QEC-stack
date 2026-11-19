"""Plot paired contrasts from the corrected trajectory records."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from adaptive_qec.analysis.causal_results import summarize


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results",type=Path,default=Path("data/suite_results.json"))
    ap.add_argument("--output-dir",type=Path,default=Path("papers/figures"))
    args=ap.parse_args()
    results=summarize(json.loads(args.results.read_text()),yaml.safe_load(Path("configs/suite.yaml").read_text()))
    fig,axes=plt.subplots(1,2,figsize=(8,3),sharey=True,layout="constrained")
    for ax,w in zip(axes,[200,2000]):
        rows=[p for p in results["panels"] if (p["distance"],p["basis"],p["window"])==(3,"Z",w)]
        for i,p in enumerate(rows):
            c=p["selective_minus"]["periodic"]
            lo,hi=np.array(c["ci95"])*100
            mid=c["mean"]*100
            ax.plot([lo,hi],[i,i],color="#35676d")
            ax.scatter([mid],[i],color="#173d44")
        ax.set_yticks(range(len(rows)),[p["scenario"].capitalize() for p in rows])
        ax.axvline(0,color="#666666",ls="--",lw=1)
        ax.set_title(f"{w:,} shots per window")
        ax.set_xlabel("LER difference (percentage points)")
    axes[0].invert_yaxis()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output_dir/"paired_comparison.pdf",metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(args.output_dir/"paired_comparison.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
