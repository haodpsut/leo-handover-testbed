#!/usr/bin/env python3
"""Hai phep do THEM cho ban camera-ready FAIR 2026.  python3 code/camera_ready_extra.py

Sinh ra tu hai nhan xet phan bien, moi phep tra loi dung mot nhan xet.

R1 (Accept 4/5): *"oversimplified binary action space ... requests evaluation on
   multi-candidate scenarios"*.
   ⇒ Do GIA cua phep rut gon. Ca hai DP deu da co san trong ma: `handover.dp_optimum` chay
   tren KHONG GIAN DAY DU (moi ve tinh nhin thay, cong NONE), con `handover_binary.solve_dp`
   chay tren khong gian nhi phan {STAY, HO toi ung vien manh nhat}. Hieu giua hai gia tri toi
   uu la gia PHAI TRA cua phep rut gon, va no may kiem duoc, khong phai uoc luong.

R2 (Weak Accept 3/5): *"lacks comprehensive comparison with existing related methods"*.
   ⇒ Them moc chuan NGANH: su kien A3 cua 3GPP, tuc tre trec (hysteresis) cong thoi gian kich
   hoat (time-to-trigger). Day la co che chuyen giao duoc dung trong thuc te, khac hai moc
   tham lam da co trong bai. Quet ca luoi (hys, TTT) de khong bi trach la chon tham so co loi.

TU KIEM: moi lich phai qua `check_feasible`; A3 voi hys=0,TTT=1 phai TRUNG max-SINR (đối chứng
dương: neu khong trung thi cai dat A3 sai).
"""
import io, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handover import (build_scenario, dp_optimum, evaluate, check_feasible,   # noqa: E402
                      greedy_max_sinr, greedy_max_elev)
import handover_binary as hb                                                   # noqa: E402

NONE = -1
loi = []


def kiem(dk, ten, ct=""):
    if not dk:
        loi.append(ten)
    print("  %s %s%s" % ("DAT " if dk else "HONG", ten, ("  [" + ct + "]") if ct else ""))


def a3_ttt(sc, hys_db=3.0, ttt=2):
    """3GPP su kien A3 + thoi gian kich hoat.

    Giu ve tinh dang phuc vu cho toi khi mot ung vien manh hon no it nhat `hys_db` dB va giu
    duoc uu the ay lien tuc `ttt` khe. Day la co che chong ping-pong kinh dien cua mang di dong,
    va la thu mot nguoi phan bien goi la "representative approach".
    """
    serv = np.full(sc.S, NONE, dtype=int)
    cur = NONE
    dem = 0          # so khe lien tiep ung vien tot nhat da vuot nguong
    ung = NONE
    for s in range(sc.S):
        vis = sc.visible[s]
        if len(vis) == 0:
            serv[s] = NONE
            cur, dem, ung = NONE, 0, NONE
            continue
        if cur == NONE or cur not in vis:
            cur = vis[int(np.argmax(sc.sinr[vis, s]))]
            dem, ung = 0, NONE
            serv[s] = cur
            continue
        khac = vis[vis != cur]
        if len(khac) == 0:
            serv[s] = cur
            dem, ung = 0, NONE
            continue
        tot = khac[int(np.argmax(sc.sinr[khac, s]))]
        if sc.sinr[tot, s] > sc.sinr[cur, s] + hys_db:
            dem = dem + 1 if tot == ung else 1
            ung = tot
            if dem >= ttt:
                cur = tot
                dem, ung = 0, NONE
        else:
            dem, ung = 0, NONE
        serv[s] = cur
    return serv


def main():
    seeds = list(range(5))
    print("== R1: GIA CUA PHEP RUT GON NHI PHAN ==")
    print("   (DP day du tren moi ve tinh nhin thay  vs  DP nhi phan STAY/HO-toi-manh-nhat)")
    print("   %-6s %11s %11s %9s %8s %8s" % ("seed", "J day du", "J nhi phan", "chenh", "%", "vis/khe"))
    gia, visn = [], []
    for sd in seeds:
        sc = build_scenario(seed=sd)
        Jf, _ = dp_optimum(sc)
        _, _, servb = hb.solve_dp(sc)
        ok, _ = check_feasible(servb, sc, verbose=False)
        kiem(ok, "lich DP nhi phan kha thi (seed %d)" % sd) if not ok else None
        Jb = evaluate(servb, sc)["J"]
        d = 100.0 * (Jb - Jf) / abs(Jf)
        v = float(np.mean([len(x) for x in sc.visible]))
        gia.append(d); visn.append(v)
        print("   %-6d %11.3f %11.3f %9.3f %7.2f%% %8.2f" % (sd, Jf, Jb, Jb - Jf, d, v))
    print("   => gia trung binh %.2f%% (sd %.2f), voi %.1f ve tinh nhin thay moi khe"
          % (np.mean(gia), np.std(gia), np.mean(visn)))
    kiem(np.mean(visn) > 5, "khong gian ung vien THAT SU rong", "%.1f" % np.mean(visn))

    print("\n== R2: MOC CHUAN NGANH, su kien A3 + TTT ==")
    sc0 = build_scenario(seed=0)
    kiem(np.array_equal(a3_ttt(sc0, hys_db=0.0, ttt=1), greedy_max_sinr(sc0)),
         "doi chung duong: A3(hys=0, TTT=1) trung khop max-SINR")

    # ⛔ HAI BAY da suyt lam con so vo nghia, ca hai deu tim ra bang cach doc run_final.py:
    #   (a) bai do gap so voi DP NHI PHAN (`handover_binary.solve_dp`), khong phai DP day du.
    #       Do bang DP day du thi moi con so A3 lech he thong so voi bang cua bai.
    #   (b) bai tach TRAIN_SEEDS=[0..4] va TEST_SEEDS=[5..9]. Chon (hys,TTT) bang cach nhin
    #       ket qua tren tap kiem la CHON TREN TAP KIEM, va se tang A3 mot loi the ma chinh
    #       sach hoc khong co. Phai chinh tren TRAIN roi danh gia tren TEST.
    TRAIN, TEST = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
    luoi = [(h, t) for h in (1.0, 2.0, 3.0, 4.0, 5.0) for t in (1, 2, 3)]

    def chay(sds, hys, ttt):
        gs, ps, hs = [], [], []
        for sd in sds:
            sc = build_scenario(seed=sd)
            _, _, servb = hb.solve_dp(sc)          # moc la DP NHI PHAN, giong bai
            Jo = evaluate(servb, sc)["J"]
            serv = a3_ttt(sc, hys, ttt)
            ok, _ = check_feasible(serv, sc, verbose=False)
            if not ok:
                loi.append("A3(%.0f,%d) sinh lich KHONG kha thi" % (hys, ttt))
            m = evaluate(serv, sc)
            gs.append(100.0 * (m["J"] - Jo) / abs(Jo))
            ps.append(m["pingpong"]); hs.append(m["handovers"])
        return np.mean(gs), np.mean(ps), np.mean(hs)

    print("   chinh tham so tren TRAIN (seed 0-4), rieng theo hai tieu chi:")
    theo_gap = min(luoi, key=lambda x: chay(TRAIN, *x)[0])
    theo_pp = min(luoi, key=lambda x: (chay(TRAIN, *x)[1], chay(TRAIN, *x)[0]))
    print("     tot nhat theo gap      : hys=%.0f dB, TTT=%d" % theo_gap)
    print("     tot nhat theo ping-pong: hys=%.0f dB, TTT=%d" % theo_pp)

    print("\n   danh gia tren TEST (seed 5-9), gap so voi DP NHI PHAN nhu trong bai:")
    print("   %-26s %9s %10s %9s" % ("phuong phap", "gap %", "ping-pong", "HO"))
    bang = []
    for ten, (h, t) in (("A3+TTT (tuned for gap)", theo_gap),
                        ("A3+TTT (tuned for p-p)", theo_pp)):
        g, p, ho = chay(TEST, h, t)
        nhan = "%s, hys=%.0f dB TTT=%d" % (ten, h, t)
        bang.append((nhan, h, t, g, p, ho))
        print("   %-26s %8.2f%% %10.1f %9.1f" % (nhan[:26], g, p, ho))

    print("\n   de doi chieu, hai moc da co trong bai (cung TEST, cung moc DP nhi phan):")
    for ten, fn in (("max-SINR", greedy_max_sinr), ("max-elev", greedy_max_elev)):
        gs, ps, hs = [], [], []
        for sd in TEST:
            sc = build_scenario(seed=sd)
            _, _, servb = hb.solve_dp(sc)
            Jo = evaluate(servb, sc)["J"]
            m = evaluate(fn(sc), sc)
            gs.append(100.0 * (m["J"] - Jo) / abs(Jo))
            ps.append(m["pingpong"]); hs.append(m["handovers"])
        print("   %-26s %8.2f%% %10.1f %9.1f" % (ten, np.mean(gs), np.mean(ps), np.mean(hs)))
        bang.append((ten, None, None, np.mean(gs), np.mean(ps), np.mean(hs)))
    print("   %-26s %8.2f%% %10.1f %9s" % ("KAN (bai, 25 to hop)", 1.158, 6.16, "--"))

    ra = {"gia_rut_gon_pct": float(np.mean(gia)), "gia_rut_gon_sd": float(np.std(gia)),
          "gia_rut_gon_max": float(np.max(gia)), "vis_per_slot": float(np.mean(visn)),
          "giao_thuc": "chinh tren seed 0-4, danh gia tren seed 5-9, gap so voi DP nhi phan",
          "bang": [{"ten": b[0], "hys": b[1], "ttt": b[2], "gap": b[3], "pp": b[4], "ho": b[5]}
                   for b in bang]}
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    io.open(os.path.join(d, "camera-ready-extra.json"), "w", encoding="utf-8").write(
        json.dumps(ra, indent=2))
    print("\n=> %s (%d loi). Da ghi results/camera-ready-extra.json"
          % ("DAT" if not loi else "CHUA DAT", len(loi)))
    for x in loi:
        print("   loi: " + x)
    return 0 if not loi else 1


if __name__ == "__main__":
    sys.exit(main())
