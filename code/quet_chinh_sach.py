#!/usr/bin/env python3
"""Quét CHÍNH SÁCH HỌC theo chế độ. KAN và MLP là ĐIỂM THAM CHIẾU, không phải chủ đề.

    python3 code/quet_chinh_sach.py --viec ofat  --nhan 16
    python3 code/quet_chinh_sach.py --viec mat2d --nhan 16
    python3 code/quet_chinh_sach.py --tu-kiem

⛔ BỐN LUẬT, và mỗi luật từ một lần sai thật ở bài này hoặc bài hội nghị:

1. ⛔⛔ **BUỘC chạy trên sol1, không được trộn máy.** Đo được ở bài này: cùng seed 0, cùng mã,
   `Σ|w|` của MLP ra **300.888793945** trên Mac (torch 2.8.0) và **302.828765869** trên sol1
   (torch 2.14.1+cpu), tức lệch **0,64%**. Đó không phải nhiễu số thực, đó là huấn luyện phân kỳ.
   KAN thì khớp tới 1,5e-5. Nên phần này là phần DUY NHẤT của bài thật sự bị ràng buộc một máy,
   và nó đủ để ràng cả bài.

2. **Chính sách học ở không gian hành động NHỊ PHÂN** (`handover_binary.rollout`: hành động là
   `{giữ, chuyển sang ứng viên mạnh nhất}`), đúng như A3. Nên mọi khoảng cách phải phân ra:

       gap(vs DP đầy đủ) = gia_rut_gon + gap(vs DP nhị phân)

   và cả ba dùng CÙNG mẫu số `|J_opt|`, nếu không thì phép cộng sai. Cổng K8 đã bắt đúng lỗi ấy
   một lần, lệch tới 0,2257 điểm.

3. **Huấn luyện trên seed TRAIN, đánh giá trên seed TEST**, và mỗi chính sách đã huấn luyện được
   đánh giá trên **mọi** kịch bản test, đúng giao thức bài hội nghị. Không trộn hai vai.

4. **Một đơn vị việc = MỘT điểm chế độ**, không phải một seed. Vì mỗi đơn vị phải dựng 40 kịch
   bản test một lần rồi dùng lại cho cả 40 chính sách; chia nhỏ hơn thì dựng lại 40 lần và tốn
   gấp bốn mươi.
"""
import argparse
import csv
import io
import json
import os
import sys
import time

# ⛔ Trước mọi import numpy/torch.
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"

import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOC = os.path.dirname(HERE)
for p in (os.path.join(GOC, "code"), os.path.join(GOC, "repo-goc", "code")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
import handover as H                                   # noqa: E402
from quet_che_do import NEO, TRUC, ten_che_do          # noqa: E402
from chay_vps import VI_DO_2D, GIA_HO_2D, xuat_xu      # noqa: E402

RES = os.path.join(GOC, "results")


def mot_che_do(dv):
    """Một điểm chế độ: huấn luyện trên mọi seed train, đánh giá trên mọi seed test."""
    import torch
    import handover_binary as HB
    from policy_train import SplineKAN, MLP, Scaler, train_policy, eval_policy, n_params
    NONE = -1

    viec, nhan, kw, seeds_tr, seeds_te = dv
    p = dict(NEO); p.update(kw)

    # --- kịch bản train, và chuyên gia nhị phân từ DP ---
    tr = []
    for s in seeds_tr:
        sc = H.build_scenario(seed=s, **p)
        _V, expert, serv_star = HB.solve_dp(sc)
        tr.append((sc, expert, serv_star))

    # --- bộ chuẩn hoá đặc trưng, lấy TRÊN TRAIN ---
    allX = []
    for sc, expert, serv_star in tr:
        n, pr = NONE, NONE
        for s in range(sc.S):
            feat, top = HB.features(sc, n, pr, s)
            if n != NONE and sc.vis[n, s] == 1 and top != NONE:
                allX.append(feat)
            m = serv_star[s]; n, pr = m, n
    if not allX:
        return []                       # chế độ suy biến: không có điểm quyết định nào
    scaler = Scaler(np.array(allX, np.float32))

    # --- kịch bản test, dựng MỘT LẦN rồi dùng lại ---
    te = []
    for s in seeds_te:
        sc = H.build_scenario(seed=s, **p)
        J_full, _ = H.dp_optimum(sc)
        _V, expert, serv_bin = HB.solve_dp(sc)
        J_bin = H.evaluate(serv_bin, sc)["J"]
        te.append((s, sc, expert, J_full, J_bin))

    hang = []
    for ten_mh, ctor in (("KAN", lambda: SplineKAN()), ("MLP", lambda: MLP(h=10))):
        for i, (sc_tr, expert_tr, serv_tr) in enumerate(tr):
            torch.manual_seed(seeds_tr[i])
            mdl = ctor()
            pol = train_policy(mdl, scaler, sc_tr, expert_tr, serv_tr)
            npar = n_params(mdl)
            for s_te, sc_te, expert_te, J_full, J_bin in te:
                r = eval_policy(sc_te, pol, expert_te, J_full)
                hang.append(dict(
                    viec=viec, che_do=nhan, chinh_sach=ten_mh, khong_gian="binary",
                    seed_train=seeds_tr[i], seed_test=s_te, so_tham_so=npar,
                    gap_pct=round(r["gap"], 6),
                    gap_vs_bin=round(100.0 * (r["J"] - J_bin) / abs(J_full), 6),
                    gia_rut_gon=round(100.0 * (J_bin - J_full) / abs(J_full), 6),
                    J=round(r["J"], 6), J_opt=round(J_full, 6), J_bin=round(J_bin, 6),
                    pingpong=r["pingpong"], handovers=r["HO"], outage=r["outage"],
                    macroF1=round(r["macroF1"], 6) if r["macroF1"] == r["macroF1"] else "",
                    N=sc_te.N, S=sc_te.S))
    return hang


def don_vi(viec, n_train, n_test):
    tr = list(range(n_train))
    te = list(range(100, 100 + n_test))
    if viec == "ofat":
        diem, thay = [], set()
        for cd in [("neo", {})] + [(ten_che_do(kw), kw) for t in TRUC for kw in TRUC[t]]:
            if cd[0] not in thay:
                thay.add(cd[0]); diem.append(cd)
        return [("ofat", nhan, kw, tr, te) for nhan, kw in diem]
    if viec == "mat2d":
        return [("mat2d", "vido=%g+lam3=%g" % (v, c), dict(station=(v, 0.0), lam3=c), tr, te)
                for v in VI_DO_2D for c in GIA_HO_2D]
    raise SystemExit("⛔ viec khong biet: %s" % viec)


def da_lam(ra):
    if not os.path.exists(ra):
        return set()
    with io.open(ra, encoding="utf-8") as f:
        return {r["che_do"] for r in csv.DictReader(f)}


def tu_kiem():
    loi = []

    def kiem(dk, ten, ct=""):
        print("  %s %s%s" % ("DAT " if dk else "HONG", ten, ("  [" + ct + "]") if ct else ""))
        if not dk:
            loi.append(ten)

    # Mot che do, 2 seed train, 2 seed test: du de kiem hinh dang va cac bat bien.
    h = mot_che_do(("tu-kiem", "neo", {}, [0, 1], [100, 101]))
    kiem(len(h) == 2 * 2 * 2, "sinh dung so dong = 2 mo hinh x 2 train x 2 test",
         "%d dong" % len(h))
    kiem({x["chinh_sach"] for x in h} == {"KAN", "MLP"}, "co ca KAN va MLP")
    kiem(all(float(x["gap_pct"]) >= -1e-9 for x in h),
         "khong chinh sach nao tot hon DP day du",
         "min %.6f" % min(float(x["gap_pct"]) for x in h))
    kiem(all(float(x["gap_vs_bin"]) >= -1e-9 for x in h),
         "khong chinh sach nhi phan nao tot hon DP NHI PHAN",
         "min %.6f" % min(float(x["gap_vs_bin"]) for x in h))
    lech = max(abs(float(x["gap_pct"]) - float(x["gia_rut_gon"]) - float(x["gap_vs_bin"]))
               for x in h)
    # ⛔ Ba cot deu `round(..., 6)`, nen tong co the lech toi 3 x 5e-7 = 1,5e-6 CHI VI lam
    #    tron, khong phai vi cong thuc sai. Nguong 1e-6 cua ban dau la chat hon ca do phan
    #    giai cua chinh du lieu, nen no bao HONG tren mot dong hoan toan dung.
    kiem(lech <= 2e-6, "phan ra gap = gia_rut_gon + gap_vs_bin (bien lam tron 2e-6)",
         "lech max %.2e" % lech)
    sp = {x["chinh_sach"]: x["so_tham_so"] for x in h}
    kiem(sp.get("KAN") == 65 and sp.get("MLP") == 211,
         "so tham so dung nhu bai hoi nghi: KAN 65, MLP 211", str(sp))
    print("\n=> %s (%d loi)" % ("DAT" if not loi else "CHUA DAT", len(loi)))
    return 1 if loi else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viec", default="ofat", choices=("ofat", "mat2d"))
    ap.add_argument("--nhan", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()
    if a.tu_kiem:
        return tu_kiem()

    os.makedirs(RES, exist_ok=True)
    ra = os.path.join(RES, "cs-%s.csv" % a.viec)
    dv = don_vi(a.viec, a.n_train, a.n_test)
    xong = da_lam(ra)
    con = [d for d in dv if d[1] not in xong]
    print("== quet_chinh_sach: viec=%s  %d che do, %d da lam, %d con lai, %d nhan =="
          % (a.viec, len(dv), len(dv) - len(con), len(con), a.nhan))
    io.open(ra + ".xuat-xu.json", "w", encoding="utf-8").write(
        json.dumps(xuat_xu(), indent=1, ensure_ascii=False))
    if not con:
        print("  khong con gi de lam (xuat xu da ghi lai).")
        return 0

    moi = not os.path.exists(ra)
    f = io.open(ra, "a", newline="", encoding="utf-8")
    w, t0, n = None, time.time(), 0
    with mp.Pool(a.nhan) as pool:
        for hang in pool.imap_unordered(mot_che_do, con, chunksize=1):
            if not hang:
                n += 1; continue
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(hang[0].keys()))
                if moi:
                    w.writeheader()
            w.writerows(hang); f.flush()
            n += 1
            d = time.time() - t0
            print("  %3d/%d che do  %.0f s  (con ~%.0f s)"
                  % (n, len(con), d, (len(con) - n) / (n / d) if n else 0), flush=True)
    f.close()
    print("  -> %s" % os.path.relpath(ra, GOC))
    return 0


if __name__ == "__main__":
    sys.exit(main())
