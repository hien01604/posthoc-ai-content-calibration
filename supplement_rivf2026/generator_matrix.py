"""Ma tran generator: 21 model x 7 ho x 5 nhiet do x 4 mode -- Revision Brief muc 2c.

    python supplement_rivf2026/generator_matrix.py

Truoc day toi bao "repo khong co ten 20 model". SAI -- chung nam trong `genai.ipynb`
(notebook sinh du lieu, van o trong repo). Script nay PARSE thang tu notebook do chu
khong go tay, nen bang luon dong bo voi cau hinh da dung de sinh du lieu.

Doi chieu cheo voi `generator_family` co that trong cac parquet da cham diem, de bao
dam ma tran khai bao khop voi du lieu thuc su ton tai.

DAU RA
------
  results/s2d_generator_models.csv   moi model: ho, pool duoc phep, co trong du lieu?
  results/s2e_generation_config.csv  nhiet do, mode, template, ty le
"""
import ast
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "artifacts" / "kaggle"
RES = HERE / "results"
NB = ROOT / "genai.ipynb"


def notebook_code(p):
    nb = json.load(open(p, encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def grab(src, name):
    """Lay literal Python cua mot bien gan o muc dinh nghia."""
    m = re.search(re.escape(name) + r"\s*=\s*", src)
    if not m:
        raise SystemExit("khong tim thay %s trong genai.ipynb" % name)
    i = m.end()
    opens = {"{": "}", "[": "]", "(": ")"}
    if src[i] not in opens:
        j = src.index("\n", i)
        return ast.literal_eval(src[i:j].split("#")[0].strip())
    close, depth = opens[src[i]], 0
    for j in range(i, len(src)):
        if src[j] in opens:
            depth += 1
        elif src[j] in "}])":
            depth -= 1
            if depth == 0:
                return ast.literal_eval(src[i:j + 1])
    raise SystemExit("khong dong ngoac cho %s" % name)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    src = notebook_code(NB)

    fam_by_key = grab(src, "FAMILY_BY_KEY")
    train_fams = grab(src, "TRAIN_FAMILIES")
    ood_fams = grab(src, "OOD_FAMILIES")
    temps = grab(src, "TEMPERATURES")
    mode_tpl = grab(src, "MODE_TEMPLATES")
    weights = grab(src, "MODE_WEIGHTS_FULL")
    quota = grab(src, "TRAIN_FAMILY_QUOTA")

    # --- ho nao THUC SU xuat hien trong du lieu da cham diem ---
    seen = set()
    for name in ["pred_val", "pred_test_ood_model"]:
        d = pd.read_parquet(ART / "artifacts_roberta_large_seed42" / ("%s.parquet" % name))
        seen |= set(d.loc[d.label == 1, "generator_family"].unique())

    rows = []
    for (provider, model), fam in sorted(fam_by_key.items(), key=lambda x: (x[1], x[0][1])):
        pool = "train/val" if fam in train_fams else ("test_ood_model" if fam in ood_fams else "?")
        rows.append(dict(provider=provider, model_id=model, family=fam,
                         split_pool=pool, family_present_in_scored_data=fam in seen))
    D = pd.DataFrame(rows)
    D.to_csv(RES / "s2d_generator_models.csv", index=False)

    cfg = []
    tot = sum(w for _, w in weights)
    MODE_DESC = {"A": "write from scratch", "B": "rewrite the human passage",
                 "C": "continue the passage", "D": "expand from bullet notes"}
    for mode, w in weights:
        cfg.append(dict(setting="generation mode", key=mode, value=MODE_DESC.get(mode, ""),
                        detail="templates: %s" % ", ".join(mode_tpl[mode]),
                        share_pct=100.0 * w / tot))
    for t in temps:
        cfg.append(dict(setting="temperature", key=str(t), value="",
                        detail="assigned by hash of (paper_id, source)",
                        share_pct=100.0 / len(temps)))
    cfg.append(dict(setting="family quota (train)", key="TRAIN_FAMILY_QUOTA",
                    value=str(quota), detail="soft cap per family", share_pct=100.0 * quota))
    C = pd.DataFrame(cfg)
    C.to_csv(RES / "s2e_generation_config.csv", index=False)

    print("=" * 84)
    print("MA TRAN GENERATOR  (parse tu genai.ipynb)")
    print("=" * 84)
    print("tong so model: %d | so ho: %d" % (len(D), D.family.nunique()))
    print("  train/val      : %d model, %d ho -> %s"
          % ((D.split_pool == "train/val").sum(), len(train_fams), sorted(train_fams)))
    print("  test_ood_model : %d model, %d ho -> %s"
          % ((D.split_pool == "test_ood_model").sum(), len(ood_fams), sorted(ood_fams)))
    print()
    print(D.to_string(index=False))
    bad = D[~D.family_present_in_scored_data]
    print("\nho khai bao nhung KHONG thay trong du lieu da cham diem: %s"
          % (sorted(set(bad.family)) if len(bad) else "khong co -- khop hoan toan"))
    print()
    print("=" * 84)
    print("CAU HINH SINH")
    print("=" * 84)
    print("nhiet do : %s  (gan bang hash, khong ngau nhien)" % temps)
    print("mode     : %s" % ", ".join("%s=%d%%" % (m, 100 * w // tot) for m, w in weights))
    print("template : %d cai / %d mode" % (sum(len(v) for v in mode_tpl.values()), len(mode_tpl)))
    print("quota ho : <=%.0f%% khoi luong train moi ho" % (100 * quota))
    print("\n-> results/s2d_generator_models.csv, s2e_generation_config.csv")


if __name__ == "__main__":
    main()
