#!/usr/bin/env python3
"""Quét CHẾ ĐỘ: khoảng cách tới tối ưu của A3 của 3GPP, theo từng trục vận hành.

Đây là dụng cụ chính của bài tạp chí. Đối tượng của bài KHÔNG phải chính sách học mà là **cơ chế
chuẩn A3**: nó cách tối ưu bao xa trong LEO, và khoảng cách ấy phụ thuộc cái gì.

    python3 code/quet_che_do.py --truc mat-do --ra results/q-mat-do.csv
    python3 code/quet_che_do.py --truc tat-ca --nhan 16
    python3 code/quet_che_do.py --tu-kiem

⛔ BỐN LUẬT MANG SANG TỪ BÀI HỘI NGHỊ, và cả bốn đều từ một lần sai thật:

1. **Khoảng cách đo so với DP ĐẦY ĐỦ**, không phải DP nhị phân. Bài hội nghị đo so với DP nhị
   phân, nên mọi con số A3 ở đó không so trực tiếp được với bài này. Tệp này ghi cột
   `goc_do = dp_full` vào từng dòng để không ai nhầm hai gốc đo.
2. **Chỉnh A3 trên seed TRAIN, đánh giá trên seed TEST.** Chỉnh trên test là tự cho điểm.
3. **Đối chứng dương ở MỌI điểm chế độ**: `hys=0, ttt=1` phải trùng khít max-SINR. Lệch một điểm
   là bộ đo hỏng, không phải một phát hiện.
4. **`OMP_NUM_THREADS=1` cho mỗi tiến trình con.** sol1 có 7 tài khoản khác dùng chung; một tiến
   trình torch mở 80 luồng thì cả máy đứng.

⚠ Và một luật của chính tệp này: nó **KHÔNG huấn luyện chính sách học**. Chính sách học là điểm
tham chiếu của bài, nằm ở `quet_chinh_sach.py` riêng, để một lần quét chế độ không kéo theo chi
phí huấn luyện và để hai thứ hỏng độc lập nhau.
"""
import argparse
import csv
import io
import itertools
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOC = os.path.dirname(HERE)
# ⛔⛔ CAY LAM VIEC PHAI THANG CAY TRONG KHO. Hai ban cua cung mot tep ton tai: `code/`
#    (dang sua) va `repo-goc/code/` (hien vat da cong bo). `sys.path.insert(0, ...)` trong
#    mot vong thi cai CHEN SAU dung TRUOC, nen thu tu liet ke quyet dinh ban nao che ban
#    nao — va mot ban CU trong kho da che ban dang sua. Dat tuong minh.
for _p in (os.path.join(GOC, "repo-goc", "code"), os.path.join(GOC, "code")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
if os.path.isdir(os.path.join(GOC, "code")):
    sys.path.remove(os.path.join(GOC, "code"))
    sys.path.insert(0, os.path.join(GOC, "code"))
import handover as H                                    # noqa: E402
from camera_ready_extra import a3_ttt                   # noqa: E402

NONE = -1
# Chế độ GỐC, đúng cấu hình bài hội nghị, để mọi trục quét quanh một điểm neo đã biết.
NEO = dict(n_planes=18, sats_per_plane=18, S=120, slot_seconds=15, fading_db=2.5,
           lam1=2.0, lam2=5.0, lam3=1.0, W_pp=2, station=(45.0, 0.0), el_min=5.0,
           sinr_th=8.0)

# Lưới A3. `hys=0, ttt=1` BẮT BUỘC có mặt: nó là đối chứng dương.
HYS = [0.0, 1.0, 2.0, 3.0, 4.0]
TTT = [1, 2, 3]

TRUC = {
    "mat-do":  [dict(n_planes=k, sats_per_plane=k) for k in (6, 9, 12, 15, 18, 24)],
    "pha-dinh": [dict(fading_db=f) for f in (0.0, 1.0, 2.5, 4.0, 6.0)],
    "gia-ho":  [dict(lam3=c) for c in (0.25, 0.5, 1.0, 2.0, 4.0)],
    "khe":     [dict(slot_seconds=d, S=int(1800 / d)) for d in (5, 15, 30, 60)],
    "vi-do":   [dict(station=(v, 0.0)) for v in (0.0, 15.0, 30.0, 45.0, 60.0, 70.0)],
}


def ten_che_do(kw):
    """Tên ngắn, đủ để truy về cấu hình. Không dùng hash: người phải đọc được."""
    if not kw:
        return "neo"
    return "+".join("%s=%s" % (k, (v if not isinstance(v, tuple) else v[0])) for k, v in sorted(kw.items()))


def mot_diem(kw, seed):
    """Một điểm chế độ, một seed. Trả về chi phí tối ưu và chi phí của từng chính sách."""
    p = dict(NEO); p.update(kw)
    sc = H.build_scenario(seed=seed, **p)
    J_opt, serv_star = H.dp_optimum(sc)
    ra = {"J_opt": J_opt, "vis_per_slot": sum(len(v) for v in sc.visible) / sc.S, "N": sc.N,
          "S": sc.S}
    # heuristic đọc được
    for ten, fn in (("max-sinr", H.greedy_max_sinr), ("max-elev", H.greedy_max_elev)):
        serv = fn(sc)
        ra[ten] = H.evaluate(serv, sc)
    # lưới A3
    for h, t in itertools.product(HYS, TTT):
        ra["a3_%g_%d" % (h, t)] = H.evaluate(a3_ttt(sc, hys_db=h, ttt=t), sc)
    return ra


def gap(J, J_opt):
    """Khoảng cách tương đối tới tối ưu, tính theo phần trăm của |J_opt|.

    ⛔ J âm (chi phí gồm `-sinr`), nên `(J - J_opt) / J_opt` ĐỔI DẤU. Chia cho `abs(J_opt)` giữ
    đúng dấu: gap > 0 nghĩa là tệ hơn tối ưu, ở mọi chế độ.

    ⚠ ĐỪNG GỌI ĐÂY LÀ "BẤT BIẾN THANG". Tôi đã viết thế trong bản đầu và nó SAI, phép tự kiểm
    in ra 1,662 so với 4,661 khi nhân đôi `lam1,lam2,lam3`. Sự thật hẹp hơn và phải nói đúng:

      - gap **không có đơn vị**, nên so được giữa các chế độ có |J_opt| khác nhau. Đó là điều
        kiện để bản đồ chế độ không trộn hai đơn vị, và đó là tất cả những gì bài cần.
      - gap **bất biến khi nhân CẢ J lẫn J_opt với một hằng số** (phép tự kiểm 3 kiểm đúng việc
        này trên công thức).
      - gap **KHÔNG bất biến khi đổi trọng số chi phí**, vì đổi trọng số là **đổi bài toán**.
        Chính vì vậy `lam3` là MỘT TRỤC QUÉT chứ không phải một chi tiết cài đặt.
    """
    return 100.0 * (J - J_opt) / abs(J_opt)


def chay_truc(ten_truc, seeds_train, seeds_test, ra_csv):
    diem = TRUC[ten_truc] if ten_truc != "neo" else [{}]
    hang = []
    t0 = time.time()
    for kw in diem:
        # --- chỉnh A3 trên TRAIN ---
        tr = [mot_diem(kw, s) for s in seeds_train]
        khoa_a3 = ["a3_%g_%d" % (h, t) for h, t in itertools.product(HYS, TTT)]
        tb_train = {k: float(np.mean([gap(r[k]["J"], r["J_opt"]) for r in tr])) for k in khoa_a3}
        tot_nhat = min(tb_train, key=tb_train.get)
        # --- đánh giá trên TEST ---
        te = [mot_diem(kw, s) for s in seeds_test]
        def tb(k, truong="J"):
            if truong == "J":
                return float(np.mean([gap(r[k]["J"], r["J_opt"]) for r in te]))
            return float(np.mean([r[k][truong] for r in te]))
        # đối chứng dương: hys=0, ttt=1 phải TRÙNG max-SINR
        lech = max(abs(r["a3_0_1"]["J"] - r["max-sinr"]["J"]) for r in te)
        for k, nhan in [(tot_nhat, "A3 tuned")] + [("max-sinr", "max-SINR"),
                                                   ("max-elev", "max-elev")]:
            hang.append(dict(truc=ten_truc, che_do=ten_che_do(kw), chinh_sach=nhan,
                             cau_hinh=(k if nhan == "A3 tuned" else ""),
                             gap_pct=round(tb(k), 4),
                             pp=round(tb(k, "pingpong"), 3),
                             ho=round(tb(k, "handovers"), 3),
                             outage=round(tb(k, "outage"), 3),
                             vis_per_slot=round(float(np.mean([r["vis_per_slot"] for r in te])), 3),
                             N=te[0]["N"], S=te[0]["S"], goc_do="dp_full",
                             n_train=len(seeds_train), n_test=len(seeds_test),
                             doi_chung_lech=round(lech, 9)))
        # dải của lưới A3: bài phải in được "chỉnh sai thì mất bao nhiêu"
        g = [tb(k) for k in khoa_a3]
        hang.append(dict(truc=ten_truc, che_do=ten_che_do(kw), chinh_sach="A3 grid range",
                         cau_hinh="%d cau hinh" % len(khoa_a3),
                         gap_pct=round(max(g) - min(g), 4), pp="", ho="", outage="",
                         vis_per_slot=round(float(np.mean([r["vis_per_slot"] for r in te])), 3),
                         N=te[0]["N"], S=te[0]["S"], goc_do="dp_full",
                         n_train=len(seeds_train), n_test=len(seeds_test),
                         doi_chung_lech=round(lech, 9)))
        print("  %-22s %-26s A3 tot nhat %-10s gap %7.3f%%  doi-chung lech %.2e"
              % (ten_truc, ten_che_do(kw), tot_nhat.replace("a3_", ""), tb(tot_nhat), lech),
              flush=True)
    if ra_csv:
        with io.open(ra_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(hang[0].keys()))
            w.writeheader(); w.writerows(hang)
        print("  -> %s (%d hang, %.1f s)" % (ra_csv, len(hang), time.time() - t0))
    return hang


def tu_kiem():
    """Đối chứng đường. Ba phép, và mỗi phép đã từng là một lỗi thật ở đâu đó."""
    loi = []

    def kiem(dk, ten, ct=""):
        print("  %s %s%s" % ("DAT " if dk else "HONG", ten, ("  [" + ct + "]") if ct else ""))
        if not dk:
            loi.append(ten)

    # 1. doi chung duong: hys=0,ttt=1 == max-SINR
    r = mot_diem({}, 0)
    d = abs(r["a3_0_1"]["J"] - r["max-sinr"]["J"])
    kiem(d < 1e-9, "doi chung duong: A3(hys=0,ttt=1) trung max-SINR", "lech %.2e" % d)

    # 2. toi uu phai KHONG TOI hon moi chinh sach
    xau = [k for k in r if isinstance(r[k], dict) and r[k]["J"] < r["J_opt"] - 1e-9]
    kiem(not xau, "khong chinh sach nao tot hon DP toi uu", ", ".join(xau[:3]))

    # 3. ⛔ Phep nay BAN DAU DAT VI LY DO SAI: no chi hoi "co tinh duoc khong", nen no xanh
    #    trong khi gap doi tu 1,662 sang 4,661 va chu thich cua toi dang khai "bat bien thang".
    #    Nay kiem DUNG hai viec, va tach chung ra.
    #    3a. bat bien khi nhan CA J lan J_opt voi mot hang so: day la tinh chat cua CONG THUC.
    for c in (0.5, 2.0, 7.3):
        g0 = gap(-100.0, -120.0)
        gc = gap(-100.0 * c, -120.0 * c)
        kiem(abs(g0 - gc) < 1e-12, "gap bat bien khi nhan ca J va J_opt voi %g" % c,
             "%.9f vs %.9f" % (g0, gc))
    #    3b. KHONG bat bien khi doi trong so chi phi, va day la DIEU DUNG: doi trong so la doi
    #        bai toan. Neu mot ngay nao do no bat bien thi `lam3` khong con la mot truc quet.
    p = dict(NEO); p.update(lam1=4.0, lam2=10.0, lam3=2.0)
    sc2 = H.build_scenario(seed=0, **p)
    J2, _ = H.dp_optimum(sc2)
    g1 = gap(r["max-sinr"]["J"], r["J_opt"])
    g2 = gap(H.evaluate(H.greedy_max_sinr(sc2), sc2)["J"], J2)
    kiem(abs(g1 - g2) > 1e-6,
         "gap DOI khi doi trong so chi phi (nen lam3 la mot truc quet that)",
         "%.3f -> %.3f" % (g1, g2))

    # 4. mat do THUA phai cho vis/khe nho hon mat do DAY
    a = mot_diem(dict(n_planes=6, sats_per_plane=6), 0)["vis_per_slot"]
    b = mot_diem(dict(n_planes=24, sats_per_plane=24), 0)["vis_per_slot"]
    kiem(a < b, "chum thua co it ve tinh nhin thay hon chum day", "%.2f < %.2f" % (a, b))

    print("\n=> %s (%d loi)" % ("DAT" if not loi else "CHUA DAT", len(loi)))
    return 1 if loi else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truc", default="neo", choices=list(TRUC) + ["neo", "tat-ca"])
    ap.add_argument("--n-train", type=int, default=5)
    ap.add_argument("--n-test", type=int, default=5)
    ap.add_argument("--ra", default=None)
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()
    if a.tu_kiem:
        return tu_kiem()
    tr = list(range(a.n_train))
    te = list(range(100, 100 + a.n_test))     # test seed tach han khoi train, khong ke nhau
    print("== Quet che do. train seed %s, test seed %s ==\n" % (tr[:3] + ["..."], te[:3] + ["..."]))
    trucs = list(TRUC) if a.truc == "tat-ca" else [a.truc]
    for t in trucs:
        ra = a.ra if len(trucs) == 1 else os.path.join(GOC, "results", "q-%s.csv" % t)
        chay_truc(t, tr, te, ra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
