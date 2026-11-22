"""Reproduce every number, table and figure of the paper from the shipped segment scores.

    python reproduce.py               # run the whole pipeline, then verify against the paper
    python reproduce.py --verify      # only compare the shipped results with paper/rivf2026.tex
    python reproduce.py --list        # list the steps
    python reproduce.py --from f5     # resume from the first step whose name contains "f5"
    python reproduce.py --only pihat  # run the steps whose name contains "pihat"

Each step is a plain script; this file only runs them in dependency order (working
directory = the script's folder) and stops at the first failure. Logs go to logs/.
Every step is seeded, so a re-run rewrites the shipped CSVs with identical values.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (script, what it produces / where it is used in the paper)
STEPS = [
    ("paper_level_tau_k.py", "rho_z (Hartung) per checkpoint and resample; static threshold"),
    ("paper_level_pihat.py", "pi-hat anchors and centring (Sec. IV-A); raw-scale Gaussian rule"),
    ("paper_level_static_vs_hartung.py", "static vs z-scale FPR by k"),
    ("paper_level_hartung_ood.py", "rho_z carried to the held-out-generator split (Table VI row 1)"),
    ("paper_level_mean_ktransfer.py", "three threshold sources (Table V, Table VI row 2)"),
    ("paper_level_rho_compare.py", "same ICC on raw vs z scale: 0.132 vs 0.296 (Sec. V-B)"),
    ("final_experiments_rivf2026/f1_pihat_ci.py", "interval coverage on human documents (Table III)"),
    ("final_experiments_rivf2026/f2_paired_bootstrap.py", "paired bootstrap (superseded by f7)"),
    ("final_experiments_rivf2026/f3_conformal_bins.py", "per-bin conformal baseline (Sec. V-D)"),
    ("final_experiments_rivf2026/f4_null_shape.py", "shape of T_k under the null, sd 0.965"),
    ("final_experiments_rivf2026/f5_deff_necessity.py", "raw/z rules with and without rho (Sec. V-A)"),
    ("final_experiments_rivf2026/f6_mixed_pi_empirical_ci.py", "coverage at pi > 0, plug-in (Table IV)"),
    ("final_experiments_rivf2026/f7_cluster_bootstrap.py", "cluster bootstrap of differences, p=0.044 (Fig. 1)"),
    ("final_experiments_rivf2026/f8_rho_sample_size.py", "calibration budget for rho_z (~200 documents)"),
    ("extra_experiments_rivf2026/exp01_calibration.py", "mixtures at known pi, k=20 (Sec. IV-A)"),
    ("extra_experiments_rivf2026/exp02_pi_by_k.py", "pi-hat by k"),
    ("extra_experiments_rivf2026/exp04_rho_sensitivity.py", "rho_z perturbed +-20% (Sec. V-B)"),
    ("extra_experiments_rivf2026/exp06_oracle.py", "per-k oracle and miss rates (Sec. V-B, VI)"),
    ("extra_experiments_rivf2026/exp08_per_checkpoint_fpr.py", "per-checkpoint FPR 0.31-1.24%"),
    ("extra_experiments_rivf2026/breakdown_per_checkpoint.py", "MAE = bias + dispersion breakdown"),
    ("revision_experiments_rivf2026/rev3_paired_miss_test.py", "paired miss test (Table VI row 3)"),
    ("revision_experiments_rivf2026/rev4_fpr_matched_power.py", "FPR-matched power, McNemar (Table VI row 4)"),
    ("supplement_rivf2026/subfield_fpr.py", "FPR by arXiv subfield (Sec. VI)"),
    ("supplement_rivf2026/submix_pihat.py", "pi-hat on real mixed documents (Supplement S8)"),
    ("supplement_rivf2026/build_supplement.py", "supplementary tables S1-S9"),
    ("make_table_ktransfer.py", "writes Table V into paper/rivf2026.tex"),
    ("make_table_mixed.py", "writes Table IV into paper/rivf2026.tex"),
    ("make_table_ood.py", "writes Table VI into paper/rivf2026.tex"),
    ("make_fig1.py", "paper/figures/fig1_fpr_curve.{pdf,png}"),
    ("make_fig2.py", "paper/figures/fig2_pipeline.{pdf,png}"),
]
VERIFY = "verify_paper_numbers_ext.py"


def run(script, log_dir):
    path = ROOT / script
    log = log_dir / (script.replace("/", "__") + ".log")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run([sys.executable, str(path)], cwd=path.parent, env=env,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - t0, log


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", action="store_true", help="only verify shipped results")
    ap.add_argument("--list", action="store_true", help="list steps and exit")
    ap.add_argument("--from", dest="start", help="resume from the first matching step")
    ap.add_argument("--only", help="run only the matching steps")
    a = ap.parse_args()

    if a.list:
        for i, (s, what) in enumerate(STEPS, 1):
            print("%2d. %-55s %s" % (i, s, what))
        return

    steps = [s for s, _ in STEPS]
    if a.start:
        idx = next((i for i, s in enumerate(steps) if a.start in s), None)
        if idx is None:
            sys.exit("no step matches %r (see --list)" % a.start)
        steps = steps[idx:]
    if a.only:
        steps = [s for s in steps if a.only in s]
    if a.verify:
        steps = []

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    total = time.time()
    for i, s in enumerate(steps, 1):
        rc, dt, log = run(s, log_dir)
        print("[%2d/%d] %-55s %6.1fs  %s" % (i, len(steps), s, dt, "ok" if rc == 0 else "FAILED"),
              flush=True)
        if rc:
            sys.exit("step failed; see %s" % log.relative_to(ROOT))
    if steps:
        print("pipeline finished in %.1f min\n" % ((time.time() - total) / 60))

    print("verifying the numbers printed in paper/rivf2026.tex ...", flush=True)
    rc = subprocess.run([sys.executable, str(ROOT / VERIFY)], cwd=ROOT,
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
