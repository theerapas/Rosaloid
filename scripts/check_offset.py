import pandas as pd, re

CSV = "GFP_single_mutation.csv"
SEQ = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
MUTCOL = "mutant"

df = pd.read_csv(CSV)
pat = re.compile(r"^[A-Z]\d+[A-Z]$")


def mismatch_count(offset):
    mism, ok = [], 0
    for s in df[MUTCOL].astype(str):
        s = s.strip()
        if not pat.match(s):
            continue
        wt, pos, mt = s[0], int(s[1:-1]), s[-1]
        idx = pos - offset
        if 0 <= idx < len(SEQ) and SEQ[idx] == wt:
            ok += 1
        else:
            if len(mism) < 10:
                here = SEQ[idx] if 0 <= idx < len(SEQ) else None
                mism.append((s, pos, offset, idx, wt, here))
    return ok, mism


# Try a range of plausible offsets
cands = list(range(-10, 11))
res = []
for off in cands:
    ok, mism = mismatch_count(off)
    res.append((off, ok, len(df), ok / len(df)))
res.sort(key=lambda x: x[3], reverse=True)

print("Offset candidates (best first):")
for off, ok, n, frac in res[:5]:
    print(f"  offset={off}: matches={ok}/{n} ({frac:.3f})")

best_off = res[0][0]
print("\nTop mismatches for best offset:")
print(mismatch_count(best_off)[1])
