#!/usr/bin/env python3
"""Cổng SỐ GỐC (G4 mục 7, SSOT). Tính lại ĐỘC LẬP mọi macro, và không chấp nhận macro mồ côi.

    python3 code/cong_so_lieu.py            # kiểm, và sinh results/claims.json
    python3 code/cong_so_lieu.py --tu-kiem  # tiêm lỗi

⛔ Vì sao không chỉ chép một "sổ khai xuất xứ": một sổ khai chép tay là một **nguồn thứ hai**,
và sổ nhà đã ghi lớp lỗi "cổng lấy kỳ vọng từ chính tệp nó đang kiểm". Ở đây cổng **tính lại
từ CSV thô bằng mã riêng**, không đọc `sinh_hinh_bang.py`, nên nó là phép đối chứng thật.

⛔ Ba hạng macro, và mỗi hạng phải xử khác nhau:
  TRUC_TIEP  tính thẳng từ CSV (lọc + gộp)        -> vào `claims.json` cho `verify_numbers.py`
  DAN_XUAT   là hàm của các macro khác             -> kiểm bằng chính công thức ấy
  NGOAI      không đến từ CSV (SHA kho, đếm văn liệu, số hình/bảng) -> phải KHAI nguồn

Macro nào không thuộc hạng nào ⇒ **mồ côi** ⇒ HỎNG. Đó là điều G7 gọi là orphan và chặn.
"""
import argparse
import csv
import io
import itertools
import json
import math
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOC = os.path.dirname(HERE)
RES = os.path.join(GOC, "results")
HYS, TTT = [0.0, 1.0, 2.0, 3.0, 4.0], [1, 2, 3]
A3 = ["a3_%g_%d" % (h, t) for h, t in itertools.product(HYS, TTT)]

# Macro KHÔNG đến từ CSV. Mỗi mục phải nói nguồn, nếu không thì nó là số từ trên trời.
NGOAI = {
    "repoSha":   "git rev-parse trong repo-goc, kiểm bởi cong_bai B?/đối chiếu ls-remote",
    "nDeXuat":   "đếm khoá \\citep giữa hai mốc trong 02-related.tex, bởi cong_van_lieu",
    # ⛔ Hai mục này KHÔNG còn là "đếm tệp trên đĩa". Đếm trên đĩa đo LẦN CHẠY TRƯỚC:
    #    `qBang` lấy trong khối macro, tức trước khi bảng được ghi, nên ở một clone sạch
    #    nó ra 0 còn trên máy tôi nó ra 8 chỉ vì bảng cũ còn đó. Nay cả hai là KHAI BÁO
    #    trong bộ sinh (`BANG`, `HINH_TIKZ`) và phép S-art dưới đây đối chiếu khai báo ấy
    #    với số tệp THỰC SỰ có, nên sai lệch không im lặng được.
    "qHinh":     "khai báo trong sinh_hinh_bang.BANG/HINH_TIKZ, đối chiếu ở phép S-art",
    "qBang":     "khai báo trong sinh_hinh_bang.BANG, đối chiếu ở phép S-art",
    "qSeedTrain": "số hạt giống test mỗi chế độ, đếm từ CSV (kiểm ở TRUC_TIEP)",
    "gLechMay":  "tính từ results/do-lech-may.json, hai tổng trọng số trên hai host",
}


def doc(ten):
    p = os.path.join(RES, ten)
    if os.path.exists(p):
        return list(csv.DictReader(io.open(p, encoding="utf-8")))
    if os.path.exists(p + ".gz"):
        import gzip
        with gzip.open(p + ".gz", "rt", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return []


def macro():
    v = {}
    for t in ("so-lieu.tex", "so-van-lieu.tex"):
        p = os.path.join(RES, t)
        if os.path.exists(p):
            v.update(dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                                     io.open(p, encoding="utf-8").read())))
    return v


def nhom(hang):
    """Gộp theo (chế độ, vai, chính sách). Cài đặt RIÊNG, không gọi sinh_hinh_bang."""
    g = {}
    for x in hang:
        g.setdefault((x["che_do"], x["vai"]), {}).setdefault(x["chinh_sach"], []).append(x)
    return g


def a3_chinh(g, cd):
    """A3 chỉnh trên train, đo trên test. Trả về cấu hình tốt nhất và các trung bình."""
    tr, te = g[(cd, "train")], g[(cd, "test")]
    tb = {k: st.mean(float(y["gap_pct"]) for y in tr[k]) for k in A3 if k in tr}
    tot = min(tb, key=tb.get)
    m = lambda k, c: st.mean(float(y[c]) for y in te[k])          # noqa: E731
    gt = [m(k, "gap_pct") for k in A3 if k in te]
    return dict(cau_hinh=tot, hys=float(tot.split("_")[1]), ttt=int(tot.split("_")[2]),
                gap=m(tot, "gap_pct"), gia=m(tot, "gia_rut_gon"), vis=m(tot, "vis_per_slot"),
                ho=m(tot, "handovers"), pp=m(tot, "pingpong"), out=m(tot, "outage"),
                khe=m(tot, "S"), msinr=m("max-sinr", "gap_pct"),
                dai=max(gt) - min(gt), te_nhat=max(gt), n=len(te[tot]),
                ktc=ktc_cum(te[tot]), hinh=so_hinh(te[tot]),
                churn=m(tot, "churn"), jopt=m(tot, "J_opt"),
                tdk=(m(tot, "J") - m(tot, "J_opt")) / m(tot, "S"),
                aa=(st.mean(float(y["gap_pct"]) for y in te["a3+a2a4"])
                    if "a3+a2a4" in te else float("nan")))


def ktc_cum(hang, cot="gap_pct"):
    """Nua be rong KTC 95% theo CUM HINH HOC. Cai dat RIENG, khong goi sinh_hinh_bang.

    ⛔ Khong duoc lay 1,96*sd/sqrt(40): 5 thuc the cung mot hinh hoc tuong quan manh, nen
       coi 40 la doc lap se BOP NHO sai so. Gop ve trung binh moi hinh hoc truoc, roi lay
       sai so chuan tren 8 so ay.
    """
    cum = {}
    for y in hang:
        cum.setdefault(y.get("kinh_do", "0"), []).append(float(y[cot]))
    cm = [st.mean(v) for v in cum.values()]
    if len(cm) < 2:
        return float("nan")
    return 1.96 * st.stdev(cm) / math.sqrt(len(cm))


def so_hinh(hang):
    return len({y.get("kinh_do", "0") for y in hang})


def tinh_lai():
    """Tính lại mọi đại lượng, trả về {ten_macro: (gia_tri, dinh_dang, mo_ta)}."""
    of, m2 = doc("vps-ofat.csv"), doc("vps-mat2d.csv")
    cs_of, cs_m2 = doc("cs-ofat.csv"), doc("cs-mat2d.csv")
    g, g2 = nhom(of), nhom(m2)
    A = {cd: a3_chinh(g, cd) for cd in {k[0] for k in g} if (cd, "train") in g and (cd, "test") in g}
    A2 = {cd: a3_chinh(g2, cd) for cd in {k[0] for k in g2}
          if (cd, "train") in g2 and (cd, "test") in g2}
    C = {}
    for x in cs_of:
        C.setdefault(x["che_do"], {}).setdefault(x["chinh_sach"], []).append(float(x["gap_pct"]))
    C = {k: {a: st.mean(b) for a, b in v.items()} for k, v in C.items()}
    C2 = {}
    for x in cs_m2:
        C2.setdefault(x["che_do"], {}).setdefault(x["chinh_sach"], []).append(float(x["gap_pct"]))
    C2 = {k: {a: st.mean(b) for a, b in v.items()} for k, v in C2.items()}

    LAT = lambda v: "neo" if v == 45.0 else "station=%.1f" % v           # noqa: E731
    COST = lambda v: "neo" if v == 1.0 else "lam3=%s" % v                # noqa: E731
    FADE = lambda v: "neo" if v == 2.5 else "fading_db=%.1f" % v         # noqa: E731
    DEN = lambda k: "neo" if k == 18 else "n_planes=%d+sats_per_plane=%d" % (k, k)   # noqa: E731

    R = {}
    def d(ten, gt, dd="%.2f", mo="", src=None):
        # src = (csv, cot, filter, agg) cho nhung dai luong DIEN DAT DUOC bang loc+gop, de
        # `verify_numbers.py` cua pipeline kiem doc lap bang cai dat cua NO. Dai luong nao
        # khong dien dat duoc (trung vi tren cac che do, tuong quan, ti so, dem) thi khong
        # co src, va chung van duoc cong nay tinh lai — chi la khong qua duoc cong kia.
        R[ten] = (gt, dd, mo, src)

    def SRC(cd, cs, cot, csv="vps-ofat.csv"):
        return (os.path.join(RES, csv), cot,
                {"che_do": cd, "vai": "test", "chinh_sach": cs}, "mean")

    # ---------- chế độ neo ----------
    neo = A["neo"]
    d("gNeo", neo["gap"], "%.2f", "A3 chinh o neo", SRC("neo", neo["cau_hinh"], "gap_pct"))
    d("gNeoSan", neo["gia"], "%.2f", "", SRC("neo", neo["cau_hinh"], "gia_rut_gon"))
    d("gNeoDu", neo["gap"] - neo["gia"], "%.2f", "gap trừ sàn")
    d("gNeoHys", neo["hys"], "%g"); d("gNeoTtt", neo["ttt"], "%d")
    d("gNeoMsinr", neo["msinr"], "%.2f", "", SRC("neo", "max-sinr", "gap_pct"))
    d("gNeoVis", neo["vis"], "%.1f", "", SRC("neo", neo["cau_hinh"], "vis_per_slot"))
    d("gNeoDai", neo["dai"], "%.1f"); d("gNeoTeNhat", neo["te_nhat"], "%.1f")
    d("qSeedTrain", neo["n"], "%d"); d("qKhe", neo["khe"], "%d")
    d("qLuoi", len(A3), "%d", "kích thước lưới (h,T)")

    # ---------- vĩ độ ----------
    d("gLatLow", A[LAT(15.0)]["gap"], "%.2f", "", SRC(LAT(15.0), A[LAT(15.0)]["cau_hinh"], "gap_pct"))
    d("gLatPolar", A[LAT(70.0)]["gap"], "%.2f", "", SRC(LAT(70.0), A[LAT(70.0)]["cau_hinh"], "gap_pct"))
    d("gLatPolarFloor", A[LAT(70.0)]["gia"]); d("gLatPolarVis", A[LAT(70.0)]["vis"], "%.2f")
    d("gLatPolarHo", A[LAT(70.0)]["ho"], "%.0f"); d("gLatPolarPp", A[LAT(70.0)]["pp"], "%.1f")
    d("gLatPolarOutPct", 100.0 * A[LAT(70.0)]["out"] / A[LAT(70.0)]["khe"], "%.0f")
    d("gLatPolarGreedy", A[LAT(70.0)]["msinr"])
    d("gLatBestVis", max(A[LAT(v)]["vis"] for v in (0., 15., 30., 45., 60., 70.)), "%.1f")
    d("gLatPolarKan", C[LAT(70.0)]["KAN"], "%.2f", "",
      (os.path.join(RES, "cs-ofat.csv"), "gap_pct", {"che_do": LAT(70.0), "chinh_sach": "KAN"}, "mean"))
    d("gLatPolarMlp", C[LAT(70.0)]["MLP"], "%.2f", "",
      (os.path.join(RES, "cs-ofat.csv"), "gap_pct", {"che_do": LAT(70.0), "chinh_sach": "MLP"}, "mean"))
    d("gLatPolarLearnBest", min(C[LAT(70.0)].values()))

    # ---------- giá chuyển giao ----------
    d("gCostLow", A[COST(0.25)]["gap"], "%.2f", "", SRC(COST(0.25), A[COST(0.25)]["cau_hinh"], "gap_pct"))
    d("gCostHigh", A[COST(4.0)]["gap"], "%.2f", "", SRC(COST(4.0), A[COST(4.0)]["cau_hinh"], "gap_pct"))
    d("gCostHighFloor", A[COST(4.0)]["gia"]); d("gCostHighGreedy", A[COST(4.0)]["msinr"])
    d("gHysDich", ", ".join("%g" % A[COST(c)]["hys"] for c in (0.25, 0.5, 1., 2., 4.)), "%s")
    d("gHysCostLo", A[COST(0.25)]["hys"], "%g"); d("gHysCostHi", A[COST(4.0)]["hys"], "%g")
    d("gCostTiQuet", round(4.0 / 0.25), "%d")
    _ts = sorted({A[c]["ttt"] for c in A})
    d("qTsaoTap", ", ".join("%d" % q for q in _ts), "%s")
    d("qTsaoMot", "yes" if _ts == [1] else "no", "%s")
    _n1 = {}
    for y in of:
        if (y["che_do"] == "neo" and y["vai"] == "test"
                and y["chinh_sach"].startswith("a3_") and y["chinh_sach"].endswith("_1")):
            _n1.setdefault(y["chinh_sach"], []).append(float(y["gap_pct"]))
    _gg = [st.mean(v) for v in _n1.values()]
    d("gNeoDaiTmot", max(_gg) - min(_gg), "%.2f")
    d("gNeoTeNhatTmot", max(_gg), "%.2f")
    sys.path.insert(0, HERE)
    from chay_vps import VI_DO_2D as _V2, GIA_HO_2D as _G2
    d("bdTrucViDo", ", ".join("%g" % v for v in _V2), "%s")
    d("bdTrucGia", ", ".join("%g" % v for v in _G2), "%s")

    # ---------- pha đinh ----------
    d("gFadeZero", A[FADE(0.0)]["gap"], "%.3f"); d("gFadeZeroFloor", A[FADE(0.0)]["gia"], "%.3f")
    d("gFadeMid", A[FADE(2.5)]["gap"], "%.2f", "", SRC(FADE(2.5), A[FADE(2.5)]["cau_hinh"], "gap_pct"))
    d("gFadeHigh", A[FADE(6.0)]["gap"], "%.2f", "", SRC(FADE(6.0), A[FADE(6.0)]["cau_hinh"], "gap_pct"))
    d("gFadeZeroKan", C[FADE(0.0)]["KAN"], "%.3f")
    d("gFadeZeroMlp", C[FADE(0.0)]["MLP"], "%.3f")

    # ---------- mật độ ----------
    md = [A[DEN(k)] for k in (6, 9, 12, 15, 18, 24)]
    d("gMongVis", A[DEN(6)]["vis"], "%.2f"); d("gMongGap", A[DEN(6)]["gap"])
    d("gMongHo", A[DEN(6)]["ho"], "%.0f"); d("gMongPp", A[DEN(6)]["pp"], "%.1f")
    d("gMatDoLo", min(x["gap"] for x in md)); d("gMatDoHi", max(x["gap"] for x in md))
    d("gMatDoDai", max(x["gap"] for x in md) - min(x["gap"] for x in md))
    d("gMatDoTiVis", max(x["vis"] for x in md) / min(x["vis"] for x in md), "%.1f")

    # ---------- điều kiện RIÊNG, và thống kê sàn ----------
    sys.path.insert(0, HERE)
    from quet_che_do import NEO as _NEO, TRUC as _TRUC, ten_che_do as _tcd
    dk, rieng = {}, []
    for ten in sorted(A):
        kw = {} if ten == "neo" else next(
            (k for tr in _TRUC for k in _TRUC[tr] if _tcd(k) == ten), None)
        if kw is None:
            continue
        p_ = dict(_NEO); p_.update(kw)
        khoa = tuple(sorted((k, str(v)) for k, v in p_.items()))
        if khoa not in dk:
            dk[khoa] = ten; rieng.append(ten)
    d("qCheDo", len(A), "%d", "số ĐIỂM đã quét trên các trục")
    d("qCheDoRieng", len(rieng), "%d", "số ĐIỀU KIỆN khác nhau")
    ti = sorted(100.0 * round(A[c]["gia"], 3) / round(A[c]["gap"], 3)
                for c in rieng if A[c]["gap"] > 1e-9)
    d("gSanTiTv", st.median(ti), "%.1f"); d("gSanTiLo", max(0.0, ti[0]), "%.1f")
    d("gSanTiHi", ti[-1], "%.1f"); d("gSanCheDo", len(ti), "%d")
    _tt = sorted(((100.0 * round(A[c]["gia"], 3) / round(A[c]["gap"], 3)), c)
                 for c in rieng if A[c]["gap"] > 1e-9)
    _nh = {"fading_db=0.0": "no fading", "fading_db=6.0": "fading at 6\\,dB",
           "fading_db=1.0": "fading at 1\\,dB", "fading_db=4.0": "fading at 4\\,dB"}
    d("gSanTiHiTen", _nh.get(_tt[-1][1], _tt[-1][1].replace("_", "\\_")), "%s")
    d("gSanTiHiSan", A[_tt[-1][1]]["gia"], "%.3f")
    d("gSanTiHiGap", A[_tt[-1][1]]["gap"], "%.3f")
    d("gFadeZeroTi", 100.0 * round(A[FADE(0.0)]["gia"], 3) / round(A[FADE(0.0)]["gap"], 3), "%.1f")
    cd = [c for c in rieng if A[c]["gap"] > 1e-9]
    xs = [A[c]["gia"] for c in cd]; ys = [A[c]["vis"] for c in cd]
    mx, my = st.mean(xs), st.mean(ys)
    d("gSanTuongQuanVis",
      sum((a - mx) * (b - my) for a, b in zip(xs, ys))
      / math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)), "%.2f")
    d("gSanLonNhatMong", A[LAT(70.0)]["gia"], "%.3f"); d("gSanNhoNhatDay", A[LAT(15.0)]["gia"], "%.3f")
    d("gVisMong", A[LAT(70.0)]["vis"], "%.1f"); d("gVisDay", A[LAT(15.0)]["vis"], "%.1f")

    # ---------- KHOANG TIN CAY, cum theo hinh hoc ----------
    d("qHinhHoc", neo["hinh"], "%d", "so hinh hoc moi vai, dem tu cot kinh_do")
    d("gNeoKtc", neo["ktc"], "%.2f")
    d("gLatLowKtc", A[LAT(15.0)]["ktc"], "%.2f")
    d("gLatPolarKtc", A[LAT(70.0)]["ktc"], "%.2f")
    d("gCostLowKtc", A[COST(0.25)]["ktc"], "%.2f")
    d("gCostHighKtc", A[COST(4.0)]["ktc"], "%.2f")

    # ---------- doi chung KHOP TAM NHIN cho san ----------
    d("gSanThuaVis", A[DEN(6)]["vis"], "%.1f")
    d("gSanThuaSan", A[DEN(6)]["gia"], "%.3f")
    d("gSanThuaTi", A[LAT(70.0)]["gia"] / A[DEN(6)]["gia"], "%.1f")
    d("gSanViDoTi", A[LAT(70.0)]["gia"] / A[LAT(15.0)]["gia"], "%.1f")
    d("gSanTiLatLow", 100.0 * A[LAT(15.0)]["gia"] / A[LAT(15.0)]["gap"], "%.1f")
    d("gSanTiLatPolar", 100.0 * A[LAT(70.0)]["gia"] / A[LAT(70.0)]["gap"], "%.1f")
    fd = [A[FADE(v)]["gap"] for v in (0., 1., 2.5, 4., 6.)]
    d("gTiPhaDinh", round(float("%.2f" % max(fd)) / float("%.2f" % min(fd)), 1), "%.1f")

    # ---------- THANG TUYET DOI, LUAN CHUYEN, A2/A4 ----------
    # ⛔ Cai dat RIENG: ti so lay tu gia tri DA LAM TRON (quy uoc §4.2), va thang tuyet doi
    #    chia cho SO KHE vi S = 1800/delta nen tong khong so duoc doc truc do dai khe.
    def _r3(a_, b_):
        return round(float("%.3f" % a_) / float("%.3f" % b_), 1)
    _vd = [(v, A[LAT(v)]) for v in (0., 15., 30., 45., 60., 70.)]
    _dinh = max(_vd, key=lambda z: z[1]["tdk"])
    d("gLatDinhViDo", _dinh[0], "%g")
    d("gLatDinhTuyetDoi", _dinh[1]["tdk"], "%.3f")
    d("gLatLowTuyetDoi", A[LAT(15.0)]["tdk"], "%.3f")
    d("gLatPolarTuyetDoi", A[LAT(70.0)]["tdk"], "%.3f")
    d("gLatGiamTuDinh",
      100.0 * (_dinh[1]["tdk"] - A[LAT(70.0)]["tdk"]) / _dinh[1]["tdk"], "%.0f")
    d("gLatTiTuyetDoi", _r3(A[LAT(70.0)]["tdk"], A[LAT(15.0)]["tdk"]), "%.1f")
    d("gLatTiMauSo", round(float("%.2f" % abs(A[LAT(15.0)]["jopt"]))
                           / float("%.2f" % abs(A[LAT(70.0)]["jopt"])), 1), "%.1f")
    d("gChurnPolar", A[LAT(70.0)]["churn"], "%.3f")
    d("gChurnThua", A[DEN(6)]["churn"], "%.3f")
    d("gChurnTi", _r3(A[LAT(70.0)]["churn"], A[DEN(6)]["churn"]), "%.1f")
    _cm = [A[DEN(k)]["churn"] for k in (6, 9, 12, 15, 18, 24)]
    d("gChurnMatDoLo", min(_cm), "%.3f"); d("gChurnMatDoHi", max(_cm), "%.3f")
    d("gChurnViDoNhi", sorted(A[LAT(v)]["churn"] for v in (0., 15., 30., 45., 60., 70.))[-2],
      "%.3f")
    d("gAaPolar", A[LAT(70.0)]["aa"], "%.2f")
    d("gAaPolarGiam", 100.0 * (A[LAT(70.0)]["gap"] - A[LAT(70.0)]["aa"])
                      / A[LAT(70.0)]["gap"], "%.0f")
    d("gAaKhacMax", max(abs(A[LAT(v)]["aa"] - A[LAT(v)]["gap"])
                        for v in (0., 15., 30., 45., 60.)), "%.3f")

    # ---------- lát 2D ----------
    vi = sorted({float(dict(kv.split("=") for kv in c.split("+"))["vido"])
                 for c in A2 if c.startswith("vido=")})
    gh = sorted({float(dict(kv.split("=") for kv in c.split("+"))["lam3"])
                 for c in A2 if c.startswith("vido=")})
    bo = [v for v in vi
          if max((A2[c]["vis"] for c in ("vido=%g+lam3=%g" % (v, q) for q in gh) if c in A2),
                 default=1) < 1e-9]
    d("bdViDoBo", bo[0] if bo else 0, "%g"); d("bdOBo", len(bo) * len(gh), "%d")
    d("bdOTruoc", len(vi) * len(gh), "%d")
    chung = [c for c in A2 if c in C2 and C2[c]]
    hieu = {c: min(C2[c].values()) - A2[c]["gap"] for c in chung}
    thang = [c for c, v in hieu.items() if v < 0]
    d("bdCells", len(chung), "%d"); d("bdWins", len(thang), "%d")
    d("bdWinPct", 100.0 * len(thang) / len(chung), "%.1f")
    d("bdBestMargin", -min(hieu.values()), "%.2f"); d("bdWorstMargin", max(hieu.values()), "%.2f")
    lv = sorted({float(dict(kv.split("=") for kv in c.split("+"))["vido"]) for c in thang})
    lg = sorted({float(dict(kv.split("=") for kv in c.split("+"))["lam3"]) for c in thang})
    d("bdLatLo", lv[0], "%g"); d("bdLatHi", lv[-1], "%g")
    d("bdCostLo", lg[0], "%g"); d("bdCostHi", lg[-1], "%g")
    # ⛔ Dem o co trung binh am KHONG phai dem phat hien: 80 o thi vai o am la tat yeu.
    #    Moi o mot KTC, cum theo hinh hoc, va chinh sach chon MOT lan theo trung binh toan o.
    a3h, csh = {}, {}
    for x in m2:
        if x["vai"] == "test" and A2.get(x["che_do"], {}).get("cau_hinh") == x["chinh_sach"]:
            a3h.setdefault((x["che_do"], x["kinh_do"]), []).append(float(x["gap_pct"]))
    for x in cs_m2:
        csh.setdefault((x["che_do"], x["chinh_sach"], x["kinh_do"]), []).append(float(x["gap_pct"]))
    nh = {}
    for c in chung:
        ten_cs = {k[1] for k in csh if k[0] == c}
        if not ten_cs:
            continue
        tot_cs = min(ten_cs, key=lambda q: st.mean(
            [w for k, vs in csh.items() if k[0] == c and k[1] == q for w in vs]))
        dd = []
        for gg in sorted({k[1] for k in a3h if k[0] == c}):
            aa, bb = a3h.get((c, gg)), csh.get((c, tot_cs, gg))
            if aa and bb:
                dd.append(st.mean(bb) - st.mean(aa))
        if len(dd) >= 2:
            nh[c] = (st.mean(dd), 1.96 * st.stdev(dd) / math.sqrt(len(dd)))
    d("bdOdoNhieu", len(nh), "%d")
    d("bdThangChac", sum(1 for mm, kk in nh.values() if mm + kk < 0), "%d")
    d("bdNhieuTV", st.median([kk for _, kk in nh.values()]), "%.2f")
    d("bdNhieuMax", max(kk for _, kk in nh.values()), "%.2f")
    d("bdBienPhatHien", st.median([kk for _, kk in nh.values()]), "%.2f")
    # Phan vi Bonferroni hai phia tren dung so o da kiem, chia cho 1,96 vi nua be rong
    # da nhan 1,96 san. Cai dat RIENG, cung cong thuc chuan.
    _bon = st.NormalDist().inv_cdf(1.0 - 0.025 / len(nh)) / 1.96
    d("bdChuanThangChac", sum(1 for mm, kk in nh.values() if mm - kk > 0), "%d")
    d("bdChuanThangBon", sum(1 for mm, kk in nh.values() if mm - kk * _bon > 0), "%d")
    # Tung kien truc rieng, de do DO LON cua loi the "chon tren tap do".
    for _a in ("KAN", "MLP"):
        _n1 = {}
        for c in chung:
            _dd = []
            for gg in sorted({k[1] for k in a3h if k[0] == c}):
                aa_, bb_ = a3h.get((c, gg)), csh.get((c, _a, gg))
                if aa_ and bb_:
                    _dd.append(st.mean(bb_) - st.mean(aa_))
            if len(_dd) >= 2:
                _n1[c] = st.mean(_dd)
        d("bdThang" + _a.title(), sum(1 for mm in _n1.values() if mm < 0), "%d")
    # TRAN LUOI: hai o "thang" la dung hai o A3 chinh ra h* = dinh luoi hysteresis.
    _hm = max(float(k.split("_")[1]) for k in A3)
    d("bdHTran", _hm, "%g")
    d("bdThangTranLuoi", sum(1 for c in thang if A2[c]["hys"] >= _hm - 1e-9), "%d")
    if len(thang) == 2:
        _kk = [dict(kv.split("=") for kv in c.split("+")) for c in thang]
        _lv2 = sorted(float(x["vido"]) for x in _kk)
        _gi = _kk[0]["lam3"]
        _mid = [v for v in vi if _lv2[0] < v < _lv2[1]]
        _cc = [c for c in ("vido=%g+lam3=%s" % (v, _gi) for v in _mid) if c in hieu]
        d("bdSoOKeGiua", len(_cc), "%d")
        if _cc:
            d("bdOKeGiuaBien", max(hieu[c] for c in _cc), "%.2f")

    # ---------- quy mô ----------
    tat = of + m2 + cs_of + cs_m2
    d("qDongTong", len(tat), "%d")
    d("qDongDayDu", sum(1 for x in tat if x.get("khong_gian") != "binary"), "%d")
    d("gSanDongMoiCauHinh", sum(1 for x in tat if x.get("khong_gian") == "binary"), "%d")
    gi = 0
    for a_, b_ in ((of, cs_of), (m2, cs_m2)):
        X = {(x["che_do"], x["seed"], x.get("kinh_do", "")) for x in a_}
        Y = {(x["che_do"], x["seed_test"], x.get("kinh_do", "")) for x in b_}
        gi += len(X & Y)
    d("qGiaoHang", gi, "%d")

    # ---------- DẪN XUẤT: hàm của macro khác, kiểm bằng chính công thức ----------
    DAN_XUAT = {
        "gLatPolarRatio": (("gLatPolar", "gLatLow"), lambda a, b: round(a / b, 1), "%.1f"),
        "gCostRatio":     (("gCostHigh", "gCostLow"), lambda a, b: round(a / b, 1), "%.1f"),
    }
    return R, DAN_XUAT


def cham(mac=None):
    loi, luu = [], []

    def kiem(dk, ten, ct=""):
        print("  %s %s%s" % ("DAT " if dk else "HONG", ten, ("  [" + ct + "]") if ct else ""))
        if not dk:
            loi.append(ten)

    m = mac if mac is not None else macro()
    R, DAN = tinh_lai()

    # ---- S-art: so HINH/BANG bai khai phai bang so tep THUC SU co tren dia ----
    # ⛔ Doi chung HAI CHIEU: bo sinh khai, dia co. Khai thieu thi bai dan mot hinh khong
    #    ton tai; dia co thua thi co hien vat khong ai dan. Phep nay bat ca hai.
    _fig = os.path.join(GOC, "figures")
    _tab = os.path.join(RES, "tables")
    _co_f = sorted(q for q in os.listdir(_fig) if q.endswith(".pdf")) if os.path.isdir(_fig) else []
    _co_b = sorted(q for q in os.listdir(_tab) if q.endswith(".tex")) if os.path.isdir(_tab) else []
    _lech = []
    if "qHinh" in m and len(_co_f) != int(m["qHinh"]):
        _lech.append("qHinh khai %s, dia co %d" % (m["qHinh"], len(_co_f)))
    if "qBang" in m and len(_co_b) != int(m["qBang"]):
        _lech.append("qBang khai %s, dia co %d" % (m["qBang"], len(_co_b)))
    kiem(not _lech, "S-art so hinh/bang bai khai khop so tep tren dia",
         "; ".join(_lech) if _lech else "%d hinh, %d bang" % (len(_co_f), len(_co_b)))

    # ---- S1 moi macro TRUC TIEP phai khop phep tinh lai doc lap ----
    lech = []
    for k, (gt, dd, _, _s) in sorted(R.items()):
        if k not in m:
            continue
        mong = (dd % gt) if dd != "%s" else gt
        if m[k] != mong:
            lech.append((k, m[k], mong))
    kiem(not lech, "S1 moi macro truc tiep khop phep tinh lai DOC LAP",
         ("lech: %s" % lech[:4]) if lech else "%d macro khop" % len([k for k in R if k in m]))

    # ---- S2 macro DAN XUAT phai bang cong thuc cua chinh no, tren SO IN RA ----
    xau = []
    for k, (vao, f, dd) in DAN.items():
        if k not in m or not all(v in m for v in vao):
            continue
        mong = dd % f(*[float(m[v]) for v in vao])
        if m[k] != mong:
            xau.append((k, m[k], mong))
    kiem(not xau, "S2 macro dan xuat bang cong thuc tren SO IN RA",
         ("lech: %s" % xau) if xau else "%d macro" % len(DAN))

    # ---- S3 KHONG macro MO COI ----
    biet = set(R) | set(DAN) | set(NGOAI)
    moc = sorted(set(m) - biet)
    kiem(not moc, "S3 khong macro mo coi (moi so deu truy duoc nguon)",
         ("mo coi: %s" % moc) if moc else "%d/%d macro co nguon" % (len(m), len(m)))

    # ---- S4 KHONG khai thua: moi muc NGOAI phai that su duoc dung ----
    thua = sorted(set(NGOAI) - set(m))
    if thua:
        print("  chu y: khai NGOAI cho macro khong con trong bai: %s" % thua); luu.append(thua)

    print("\n=> %s (%d loi, %d chu y)" % ("DAT" if not loi else "CHUA DAT", len(loi), len(luu)))
    return loi, R, DAN


def ghi_claims(R, m):
    """Sinh claims.json cho `verify_numbers.py verify` — chỉ các mục lọc+gộp được."""
    ds = []
    for k, (gt, dd, mo, src) in sorted(R.items()):
        if k in m and src:
            csv_, cot, flt, agg = src
            ds.append({"id": k, "paper_value": m[k], "csv": csv_, "column": cot,
                       "filter": flt, "agg": agg})
    p = os.path.join(RES, "claims.json")
    io.open(p, "w", encoding="utf-8").write(json.dumps(ds, indent=1, ensure_ascii=False))
    print("  -> %s (%d muc)" % (os.path.relpath(p, GOC), len(ds)))


def tu_kiem():
    m0 = macro()
    print("== Ban GOC ==")
    l, R, DAN = cham(m0)
    if l:
        sys.exit("⛔ ban GOC da hong")
    ca = [("S1: doi mot macro truc tiep", {**m0, "gNeo": "9.99"}),
          ("S2: doi mot macro dan xuat", {**m0, "gLatPolarRatio": "9.9"}),
          ("S3: them mot macro mo coi", {**m0, "gBiaRa": "1.23"})]
    hong = []
    for ten, mm in ca:
        print("\n== %s ==" % ten)
        if cham(mm)[0]:
            print("  => cong BAT duoc")
        else:
            print("  => ⛔ cong KHONG bat"); hong.append(ten)
    print("\n=> %s (%d/%d ca)" % ("DAT" if not hong else "CHUA DAT", len(ca) - len(hong), len(ca)))
    return 1 if hong else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()
    if a.tu_kiem:
        return tu_kiem()
    print("== Cong so goc (SSOT) ==\n")
    l, R, _ = cham()
    if not l:
        ghi_claims(R, macro())
    return 1 if l else 0


if __name__ == "__main__":
    sys.exit(main())
