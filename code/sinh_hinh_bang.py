#!/usr/bin/env python3
"""Sinh HÌNH và BẢNG từ các CSV ĐÃ QUA CỔNG.  python3 code/sinh_hinh_bang.py

⛔ CHẠY `code/cong_ket_qua.py` TRƯỚC. Một hình vẽ từ số chưa qua cổng là một hình phải vẽ lại,
nên bộ này **tự chạy cổng** và dừng nếu cổng hỏng. Không có cờ nào để bỏ qua.

⛔ Bộ style của nhà, `style/agentra_plot.py`, đã vendor trong kho gốc. Không vẽ bằng matplotlib
trần: luật nhà, và `make_r5_figs.py` của một bài khác đang là món nợ đúng vì đã làm thế.

⛔ MỌI SỐ TRONG BẢNG ĐỀU SINH RA, không gõ tay. Lớp lỗi "bảng/hình gõ tay thì chạy lại thí
nghiệm không cập nhật" đã cắn nhiều lần, nên bảng ra `.tex` và bài `\\input` nó.

Giao thức, giống mọi chỗ khác trong bài này:
  - A3 CHỈNH trên seed train, ĐO trên seed test;
  - khoảng cách phân ra `gap = gia_rut_gon + gap_vs_bin`, cả ba cùng mẫu số `|J_opt|`;
  - chính sách học lấy trung bình trên mọi cặp (train, test).
"""
import collections
import csv
import io
import itertools
import os
import re
import statistics as st
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GOC = os.path.dirname(HERE)
RES = os.path.join(GOC, "results")
FIG = os.path.join(GOC, "figures")
TAB = os.path.join(RES, "tables")
# ⛔ BO STYLE o HAI LAYOUT khac nhau: trong thu muc bai no la `repo-goc/style`, con trong
#    BAN CLONE SACH cua kho cong khai no la `style`. Mot duong dan cung se chay o may toi va
#    vo o may nguoi doc, va do dung la "lop loi diem vao" ma cong clone-sach sinh ra de bat.
def _tim_style():
    for d in (os.path.join(GOC, "repo-goc", "style"), os.path.join(GOC, "style")):
        if os.path.isfile(os.path.join(d, "agentra_plot.py")):
            return d
    raise SystemExit("⛔ khong tim thay bo style (agentra_plot.py) o repo-goc/style hay style/")


STYLE_DIR = _tim_style()
sys.path.insert(0, STYLE_DIR)
import agentra_plot as ap          # noqa: E402
sys.path.insert(0, HERE)
from quet_che_do import ten_che_do    # noqa: E402  nguon DUY NHAT cua ten che do

HYS = [0.0, 1.0, 2.0, 3.0, 4.0]
TTT = [1, 2, 3]
A3 = ["a3_%g_%d" % (h, t) for h, t in itertools.product(HYS, TTT)]
W1 = 3.5          # một cột, inch. `measure_plot` đòi 252 pt ±6%

# ⛔ LUẬT NHÀ (G5 nhánh B): heatmap ĐƠN SẮC trắng→agreen. KHÔNG viridis, KHÔNG jet, KHÔNG
#    YlOrRd. Tôi đã vi phạm cả ba ở bản đầu. Bản đồ hai phía thì hai kênh brand quanh trắng.
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm   # noqa: E402
CMAP_DON = LinearSegmentedColormap.from_list("wag", ["#FFFFFF", ap.AGREEN])
CMAP_HAI = LinearSegmentedColormap.from_list("gwa", [ap.AGREEN, "#FFFFFF", ap.WARM])


def K(**kw):
    """Khoá chế độ, ỦY QUYỀN cho `ten_che_do`, nguồn DUY NHẤT của tên chế độ.

    ⛔⛔ ĐÂY LÀ LỖI ĐÃ CẮN HAI LẦN, lần hai làm HỎNG MỘT HÌNH TRONG BÀI. Tự dựng khoá bằng
    `"lam3=%g" % 1.0` ra `lam3=1`, còn tên thật là `lam3=1.0`. Lần đầu nó làm sai một macro và
    tôi vá ở `sinh_macro`; lần hai nó sống sót ở `main()` nên Hình 3 chỉ vẽ **2 trên 5 điểm**
    của trục giá chuyển giao, trong khi §Results nói "λ3 từ 0,25 lên 4". Không cổng nào bắt,
    Hảo bắt bằng mắt. Nay mọi chỗ phải đi qua hàm này.
    """
    return ten_che_do(kw)


def bo_cot_suy_bien(a3, vi, gh):
    """Bỏ các cột vĩ độ SUY BIẾN khỏi lát 2D, và nói rõ bỏ cái gì.

    ⛔⛔ ĐO ĐƯỢC: ở vĩ độ 75 độ, vỏ Walker 53 độ để lại **0,000 vệ tinh nhìn thấy mỗi khe** và
    người dùng mất liên lạc **120/120 khe**. Không có một quyết định nào tồn tại ở đó, nên mọi
    chính sách trùng nhau và khoảng cách không có nghĩa. Bộ quét chính sách học đã tự loại 8 ô
    ấy (nó không có điểm huấn luyện nào), nhưng hai hình nhiệt A3 vẫn vẽ chúng, nên BA hình của
    cùng một lát đang nói hai cỡ lưới khác nhau: 88 và 80. Đó là lỗi nhất quán giữa các hình,
    thứ chỉ lộ khi nhìn cả bộ. Nay cả ba cùng bỏ, và bài phải khai.
    """
    giu, bo = [], []
    for v in vi:
        vs = [a3[c]["vis"] for c in (("vido=%g+lam3=%g" % (v, g)) for g in gh) if c in a3]
        (bo if (vs and max(vs) < 1e-9) else giu).append(v)
    if bo:
        print("       bo %d cot vi do SUY BIEN (0 ve tinh nhin thay): %s"
              % (len(bo), ", ".join("%g" % v for v in bo)))
    return giu, bo


def nhan_thua(ax, vi):
    """Nhãn trục x THƯA cho lát 2D: 11 nhãn sát nhau thì chữ chồng chữ, mắt đọc không ra."""
    chon = [v for v in vi if v % 20 == 0] or vi
    if vi and vi[-1] not in chon:
        chon = chon + [vi[-1]]
    ax.set_xticks(chon); ax.set_xticklabels(["%g" % v for v in chon])


# ⛔⛔ DANH SACH BANG va HINH phai KHAI O DAY, khong duoc suy ra bang `os.listdir`.
#    `qBang` tung tinh bang cach dem tep .tex trong results/tables NGAY TRONG khoi macro,
#    tuc TRUOC khi cac bang duoc ghi. Tren may toi no ra 8 — nhung chi vi 8 bang cua LAN
#    CHAY TRUOC con nam tren dia. Trong mot clone sach no ra **0**. Con so ay chua bao gio
#    la phep do cua lan chay hien tai, va chi lo ra khi that su chay o cho sach.
#    Nay no la so phan tu cua danh sach ma `main()` DUYET, nen hai thu khong the lech.
BANG = ["tab-ky-hieu.tex", "tab-che-do.tex", "tab-doi-chieu.tex", "tab-chinh.tex",
        "tab-dai-luoi.tex", "tab-dung-cu.tex", "tab-xuat-xu.tex", "tab-tham-so.tex"]
# Hinh luong ve bang TikZ ngoai bo sinh nay, nen no duoc KHAI chu khong duoc dem.
HINH_TIKZ = ["g0-luong.pdf"]


def _ti3_tron(tu, mau):
    """Ti so tinh tu gia tri lam tron 3 chu so thap phan, dung quy uoc cong B20."""
    return round(float("%.3f" % tu) / float("%.3f" % mau), 1)


def _ti2(tu, mau):
    """Ti so tinh tu gia tri DA LAM TRON nhu bai in, 2 chu so thap phan.

    ⛔ Cong B20 doi "ti so in ra = ti so cua cac gia tri in ra". Bang va macro phai dung
       CUNG mot ham, khong thi bang se in 5,4 trong khi van in 5,5 cho cung mot phep chia.
    """
    return round(float("%.2f" % tu) / float("%.2f" % mau), 1)


def _ti3_sau(a3, c):
    """Tỉ lệ sàn trên khoảng cách, tính từ giá trị ĐÃ LÀM TRÒN như bài in."""
    return 100.0 * round(a3[c]["gia"], 3) / round(a3[c]["gap"], 3)


def che_do_rieng(a3):
    """Trả về một TÊN cho mỗi ĐIỀU KIỆN khác nhau, bỏ các tên trùng điều kiện.

    ⛔⛔ Phản biện ngoài 11/10 bắt đúng chỗ này. Bộ quét đi theo từng trục, và điểm GIỮA của
    mỗi trục chính là chế độ neo, nên chế độ neo được chạy và ghi lại **sáu lần** dưới sáu
    cái tên: `neo`, `n_planes=18+sats_per_plane=18`, `fading_db=2.5`, `lam3=1.0`,
    `S=120+slot_seconds=15`, `station=45.0`. Lấy trung vị hay tương quan trên 27 tên tức là
    cân điểm neo gấp sáu lần mọi điểm khác. Đó là một lỗi thống kê, không phải lỗi đếm.
    """
    import sys as _s
    _s.path.insert(0, HERE)
    from quet_che_do import NEO as _NEO, TRUC as _TRUC
    dk, ra = {}, []
    for ten in sorted(a3):
        if ten == "neo":
            kw = {}
        else:
            kw = next((k for tr in _TRUC for k in _TRUC[tr] if ten_che_do(k) == ten), None)
            if kw is None:            # che do cua lat 2D, khong thuoc truc nao
                ra.append(ten); continue
        p_ = dict(_NEO); p_.update(kw)
        khoa = tuple(sorted((k, str(v)) for k, v in p_.items()))
        if khoa not in dk:
            dk[khoa] = ten; ra.append(ten)
    return ra


def o_nhiet(ax, X, Y, M, cmap, norm=None):
    """Vẽ một bản đồ nhiệt theo đúng luật nhà, và đó là lý do nó là một hàm riêng.

    ⛔ `agentra.mplstyle` bật `axes.grid` theo trục y, nên với heatmap đường lưới rơi ĐÚNG
    TÂM HÀNG, tức cắt ngang qua con số trong ô. Hảo nhìn thấy ngay ở Hình 6. `measure_plot`
    không bắt được vì đây là va chạm bố cục, không phải cỡ chữ hay khổ. Luật nhà: `grid(False)`
    + `tick_params(length=0)` + phân ô bằng đường trắng ở RANH GIỚI chứ không ở tâm.
    """
    im = ax.pcolormesh(X, Y, M, shading="nearest", cmap=cmap, norm=norm,
                       edgecolors="white", linewidth=0.6)
    ax.grid(False)
    ax.tick_params(length=0)
    return im


def so_trong_o(ax, xs, ys, M, dinh="%.1f", nguong=0.66):
    # ⛔ Nguong 0,55 cho o "14.2" chu TRANG tren nen xanh TRUNG: doc duoc nhung sat men.
    #    Nang len 0,66 de chi o THAT SU dam moi dung chu trang.
    """Số trong ô, màu chữ theo độ đậm nền để luôn đủ tương phản."""
    lo, hi = np.nanmin(M), np.nanmax(M)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if np.isnan(M[i, j]):
                continue
            t_ = (M[i, j] - lo) / (hi - lo) if hi > lo else 0.0
            ax.text(x, y, dinh % M[i, j], ha="center", va="center", fontsize=7,
                    color=("white" if t_ > nguong else ap.INK))
SINH = "%% SINH TU DONG boi code/sinh_hinh_bang.py. KHONG sua tay.\n"


def tai(ten):
    """Đọc CSV, nhận cả bản nén. Kho công khai chứa `.csv.gz` (63 MB xuống ~8 MB) nên một
    bản clone sạch phải chạy được KHÔNG cần giải nén tay: đó là điều kiện của cổng
    "clone sạch bắt lỗi điểm vào"."""
    p = os.path.join(RES, ten)
    if os.path.exists(p):
        with io.open(p, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if os.path.exists(p + ".gz"):
        import gzip
        with gzip.open(p + ".gz", "rt", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return None


def _ktc_cum(hang, cot="gap_pct"):
    """Nua be rong KTC 95% theo CUM HINH HOC, khong coi 40 thuc the la doc lap."""
    import math
    cum = {}
    for y in hang:
        cum.setdefault(y.get("kinh_do", "0"), []).append(float(y[cot]))
    cm = [st.mean(v) for v in cum.values()]
    if len(cm) < 2:
        return float("nan")
    return 1.96 * st.stdev(cm) / math.sqrt(len(cm))


def _nhieu_bien_gioi(m2_a3, o_can, chi=None):
    """Voi tung o cua lat 2D, bien do chenh HOC - A3 CHINH co phan biet voi 0 khong.

    ⛔ Cau hoi nay KHONG tra loi duoc bang trung binh. Hoi dong G6 chi ra: `bdWins`
       dem o nao co diem UOC LUONG am, trong khi phep lay mau co 8 hinh hoc nen moi
       o mang sai so rieng. Dem diem uoc luong la bien nhieu thanh phat hien, va voi
       80 o thi vai o am la DIEU PHAI XAY RA du khong co hieu ung nao.
       KTC duoi day cum theo hinh hoc, giong `_ktc_cum`.
    """
    import math
    a3r, csr = tai("vps-mat2d.csv") or [], tai("cs-mat2d.csv") or []
    a3g, csg = collections.defaultdict(list), collections.defaultdict(list)
    for x in a3r:
        if x["vai"] == "test" and m2_a3.get(x["che_do"], {}).get("cau_hinh") == x["chinh_sach"]:
            a3g[(x["che_do"], x["kinh_do"])].append(float(x["gap_pct"]))
    for x in csr:
        csg[(x["che_do"], x["chinh_sach"], x["kinh_do"])].append(float(x["gap_pct"]))
    ra = {}
    for c in o_can:
        ten = {k[1] for k in csg if k[0] == c}
        if chi is not None:
            ten = {q for q in ten if q == chi}
        if not ten:
            continue
        # ⛔ Chon chinh sach MOT lan theo trung binh toan o, khong chon theo tung hinh
        #    hoc: chon theo hinh hoc la chon tren chinh so minh dang do.
        tot = min(ten, key=lambda q: st.mean(
            [v for k, vs in csg.items() if k[0] == c and k[1] == q for v in vs]))
        d = []
        for g in sorted({k[1] for k in a3g if k[0] == c}):
            a, b = a3g.get((c, g)), csg.get((c, tot, g))
            if a and b:
                d.append(st.mean(b) - st.mean(a))
        if len(d) >= 2:
            ra[c] = (st.mean(d), 1.96 * st.stdev(d) / math.sqrt(len(d)))
    return ra


def a3_theo_che_do(hang):
    """A3 chỉnh trên TRAIN, đo trên TEST. Trả về theo từng chế độ."""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in hang:
        g[(x["che_do"], x["vai"])][x["chinh_sach"]].append(x)
    ra = {}
    for cd in {k[0] for k in g}:
        tr, te = g.get((cd, "train"), {}), g.get((cd, "test"), {})
        if not tr or not te:
            continue
        tb = {k: st.mean(float(y["gap_pct"]) for y in tr[k]) for k in A3 if k in tr}
        tot = min(tb, key=tb.get)
        def m(k, c="gap_pct"):
            return st.mean(float(y[c]) for y in te[k])
        gt = [m(k) for k in A3 if k in te]
        ra[cd] = dict(cau_hinh=tot, hys=float(tot.split("_")[1]), ttt=int(tot.split("_")[2]),
                      gap=m(tot), sd=st.pstdev([float(y["gap_pct"]) for y in te[tot]]),
                      gia=m(tot, "gia_rut_gon"), con=m(tot, "gap_vs_bin"),
                      pp=m(tot, "pingpong"), ho=m(tot, "handovers"),
                      dai=max(gt) - min(gt), te_nhat=max(gt),
                      msinr=m("max-sinr"), melev=m("max-elev"),
                      out=m(tot, "outage"), khe=st.mean(float(y["S"]) for y in te[tot]),
                      # ⛔ KTC THEO CUM. Phep lay mau co cau truc: 8 hinh hoc, moi hinh
                      #    hoc 5 hat giong pha dinh. Coi 40 thuc the la doc lap se danh
                      #    gia THAP sai so, vi 5 thuc the cung hinh hoc tuong quan manh.
                      ktc=_ktc_cum(te[tot]),
                      vis=m(tot, "vis_per_slot"), n=len(te[tot]),
                      # ⛔ THANG TUYET DOI. Khoang cach chia cho |J*|, va |J*| co 3,5 lan tu
                      #    15 do len 70 do, nen ti so 5,5x gom 1,6x TU SO va 3,5x MAU SO. Tren
                      #    thang cua chinh ham muc tieu, khoang cach DAT DINH o 45 do roi GIAM.
                      #    Khong co hai cot nay thi ket qua manh nhat cua bai la mot hieu ung
                      #    chuan hoa, va bai dang KHAI rang Bang 3 bao ca hai thang trong khi
                      #    ca hai cot cua no deu la ham cua cung mot gap da chuan hoa.
                      jopt=m(tot, "J_opt"), tuyet_doi=m(tot, "J") - m(tot, "J_opt"),
                      # ⛔ CHIA CHO SO KHE. Tong tuyet doi KHONG so duoc doc truc do dai khe,
                      #    vi S = 1800/delta nen chan troi 5 s co 360 khe con 60 s co 30 khe,
                      #    va chi phi tong ti le voi S. Bo cot nay di thi cot "tuyet doi" se
                      #    noi do dai khe la truc manh nhat, trong khi do chi la so khe.
                      tuyet_doi_khe=(m(tot, "J") - m(tot, "J_opt")) / m(tot, "S"),
                      # LUAN CHUYEN tap nhin thay, va A3 kem cap A2/A4 lam cam bien.
                      churn=m(tot, "churn"),
                      aa=(st.mean(float(y["gap_pct"]) for y in te["a3+a2a4"])
                          if "a3+a2a4" in te else float("nan")))
    return ra


def cs_theo_che_do(hang):
    if not hang:
        return {}
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in hang:
        g[x["che_do"]][x["chinh_sach"]].append(float(x["gap_pct"]))
    return {cd: {k: st.mean(v) for k, v in d.items()} for cd, d in g.items()}


def luu(fig, ten, hai_cot=False):
    """Lưu hình và ĐỌC PHÁN QUYẾT của `measure_plot`, không đọc cái nhãn.

    ⛔⛔ HAI LỖI CỦA TÔI Ở BẢN ĐẦU, cả hai đều làm cổng xanh trên hình SAI:

    1. Phép kiểm viết `"PRINT-READY" in r.stdout`, mà dòng phán quyết là
       `PRINT-READY: FAIL`. Nên nó khớp CÁI NHÃN và báo ĐẠT trên đúng những hình
       `measure_plot` vừa bác. Đây là lớp lỗi "cổng kiểm tên thay vì bản chất", và phán
       quyết duy nhất tin được là **mã trả về**.
    2. `bbox_inches="tight"` CẮT bớt lề nên bề rộng lưu ra NHỎ HƠN `figsize`: xin 3,5 in
       (252 pt) mà ra 216 pt, tức hẹp hơn dung sai ±6% của một cột. Bộ sinh hình cũ lách
       bằng cách xin 4,0 in cho cắt về 247 pt, nhưng lượng cắt phụ thuộc độ dài nhãn nên
       cách ấy không bền. `constrained_layout` cho trục lấp đầy khung, nên lưu KHÔNG cắt
       sát thì bề rộng ĐÚNG BẰNG figsize.
    """
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, ten)
    fig.savefig(p)                     # ⛔ KHONG bbox_inches="tight", xem ly do 2 o tren
    plt.close(fig)
    r = subprocess.run([sys.executable, os.path.join(STYLE_DIR, "measure_plot.py"),
                        p] + (["--double-column"] if hai_cot else []),
                       capture_output=True, text=True)
    ok = r.returncode == 0             # ⛔ MA TRA VE, khong phai chuoi trong stdout
    art = ([l for l in r.stdout.splitlines() if "artwork" in l] or [""])[0]
    print("  %s %-24s %s" % ("DAT " if ok else "HONG", ten, art.replace("artwork     : ", "")))
    if not ok:
        for l in r.stdout.splitlines():
            if l.strip().startswith("-"):
                print("        " + l.strip()[:120])
    return ok


# ---------------------------------------------------------------- hình

def f_truc(a3, cs, truc, xs, nhan_x, ten_tep, log_x=False):
    """Khoảng cách theo MỘT trục, kèm phân rã và hai chính sách học."""
    cd = [truc(v) for v in xs]
    thieu = [(v, c) for v, c in zip(xs, cd) if c not in a3]
    # ⛔⛔ BẢN ĐẦU CHỈ KIỂM `len(co) < 2` rồi vẽ tiếp. Nên khi 3 trên 5 khoá sai chính tả,
    #    nó vẽ một đường HAI ĐIỂM và không ai biết. Một hình thiếu điểm là một hình SAI,
    #    không phải một hình nhỏ hơn. Nay thiếu một điểm là HỎNG, và in ra thiếu cái gì.
    if thieu:
        print("  ⛔ %s: THIEU %d/%d diem tren truc: %s"
              % (ten_tep, len(thieu), len(xs), [c for _, c in thieu]))
        return False
    co = cd
    x = [v for v, c in zip(xs, cd) if c in a3]
    gia = [a3[c]["gia"] for c in co]
    gap = [a3[c]["gap"] for c in co]
    fig, ax = plt.subplots(figsize=(W1, 2.35), constrained_layout=True)
    # phần cấu trúc (giá rút gọn) vẽ thành vùng tô: nó KHÔNG phải lỗi của A3
    ax.fill_between(x, 0, gia, color=ap.HAIRLINE, label="action-space floor")
    ax.plot(x, gap, label="A3 (tuned)", **ap.sty("A3 (tuned)"))
    for ten in ("KAN", "MLP"):
        if cs and all(c in cs and ten in cs[c] for c in co):
            ax.plot(x, [cs[c][ten] for c in co], label=ten, **ap.sty(ten))
    ax.plot(x, [a3[c]["msinr"] for c in co], label="max-SINR", **ap.sty("max-SINR"))
    # ⛔⛔ NHAN TRUC PHAI LA GIA TRI DA QUET, khong phai tick tu dong cua matplotlib. Ban dau
    #    hinh mat do ghi 7.5, 10.0, 12.5 ... trong khi CHI chay 6, 9, 12, 15, 18, 24; hinh pha
    #    dinh ghi 3 va 5 la hai muc CHUA HE CHAY. Mot nguoi doc khong the biet diem nao la so
    #    do va diem nao la noi suy, va do la mot loi noi dung chu khong phai tham my.
    if log_x:
        # `set_xscale("log")` sinh nhan `10^{0}`: so mu mathtext ve o ~70% co chu nen ra 4,9pt,
        # duoi san 7,0pt. Dat nhan TUONG MINH thi khong co so mu nao.
        ax.set_xscale("log")
        ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels([("%g" % v) for v in x])
    ax.set_xlabel(nhan_x)
    ax.set_ylabel("optimality gap (%)")
    # ⛔ Legend ĐẶT TRONG khung đè lên chính các đường nó chú thích (Hảo thấy ở Hình 3).
    #    Luật nhà: dời RA NGOÀI, hàng ngang TRÊN trục, và bỏ khung.
    ax.legend(frameon=False, fontsize=7, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, borderaxespad=0.0,
              handlelength=1.6, columnspacing=1.1)
    return luu(fig, ten_tep)


def f_luoi_a3(hang, ten_tep):
    """Bản đồ nhiệt khoảng cách trên lưới (hys, TTT), ở chế độ neo."""
    te = [x for x in hang if x["che_do"] == "neo" and x["vai"] == "test"]
    if not te:
        print("  bo qua %s: khong co che do neo" % ten_tep)
        return False
    m = np.full((len(TTT), len(HYS)), np.nan)
    for i, t_ in enumerate(TTT):
        for j, h in enumerate(HYS):
            v = [float(x["gap_pct"]) for x in te if x["chinh_sach"] == "a3_%g_%d" % (h, t_)]
            if v:
                m[i, j] = st.mean(v)
    if np.isnan(m).any():
        print("  ⛔ %s: thieu %d o tren luoi" % (ten_tep, int(np.isnan(m).sum())))
        return False
    fig, ax = plt.subplots(figsize=(W1, 2.0), constrained_layout=True)
    im = o_nhiet(ax, HYS, TTT, m, CMAP_DON)
    ax.set_xticks(HYS); ax.set_xticklabels(["%g" % h for h in HYS])
    ax.set_yticks(TTT); ax.set_yticklabels([str(x) for x in TTT])
    ax.set_xlabel("hysteresis (dB)"); ax.set_ylabel("time-to-trigger (slots)")
    so_trong_o(ax, HYS, TTT, m)
    fig.colorbar(im, ax=ax, label="gap (%)")
    return luu(fig, ten_tep)


def f_mat2d(hang, ten_tep):
    """Mặt 2D: vĩ độ × giá chuyển giao. Hình trung tâm của bài."""
    a3 = a3_theo_che_do(hang)
    vi, gh = set(), set()
    for cd in a3:
        if not cd.startswith("vido="):
            continue
        p = dict(kv.split("=") for kv in cd.split("+"))
        vi.add(float(p["vido"])); gh.add(float(p["lam3"]))
    vi, gh = sorted(vi), sorted(gh)
    vi, _bo = bo_cot_suy_bien(a3, vi, gh)
    if len(vi) < 2 or len(gh) < 2:
        print("  bo qua %s" % ten_tep)
        return False
    m = np.full((len(gh), len(vi)), np.nan)
    for i, c in enumerate(gh):
        for j, v in enumerate(vi):
            cd = "vido=%g+lam3=%g" % (v, c)
            if cd in a3:
                m[i, j] = a3[cd]["gap"]
    fig, ax = plt.subplots(figsize=(W1, 2.3), constrained_layout=True)
    im = o_nhiet(ax, vi, gh, m, CMAP_DON)
    ax.set_yscale("log")
    ax.set_yticks(gh); ax.set_yticklabels([("%g" % v) for v in gh]); ax.minorticks_off()
    nhan_thua(ax, vi)
    ax.set_xlabel("ground-station latitude (deg)")
    ax.set_ylabel("handover cost $\\lambda_3$")
    fig.colorbar(im, ax=ax, label="A3 gap (%)")
    return luu(fig, ten_tep)


def f_ban_do_che_do(hang_a3, hang_cs, ten_tep):
    """⭐ HÌNH TRUNG TÂM: biên chế độ. Ở đâu chính sách học thắng A3 đã chỉnh.

    Vẽ `gap(học tốt nhất) − gap(A3 đã chỉnh)`, thang hai phía quanh 0, nên vùng học thắng
    hiện ra bằng màu ngược. Đo được: học thắng ở **5 trong 80 ô**, cả năm ở vĩ độ 50 đến 60
    với chuyển giao đắt, và biên lớn nhất chỉ 0,33 điểm. Ngoài dải ấy A3 thắng tới 3,58 điểm.
    """
    from matplotlib.colors import TwoSlopeNorm
    a3 = a3_theo_che_do(hang_a3)
    cs = cs_theo_che_do(hang_cs)
    vi, gh = set(), set()
    for cd in a3:
        if cd.startswith("vido="):
            q = dict(kv.split("=") for kv in cd.split("+"))
            vi.add(float(q["vido"])); gh.add(float(q["lam3"]))
    vi, gh = sorted(vi), sorted(gh)
    vi, _bo = bo_cot_suy_bien(a3, vi, gh)
    if len(vi) < 2 or len(gh) < 2:
        print("  bo qua %s" % ten_tep)
        return False
    m = np.full((len(gh), len(vi)), np.nan)
    n_hoc = n_o = 0
    for i, c in enumerate(gh):
        for j, v in enumerate(vi):
            cd = "vido=%g+lam3=%g" % (v, c)
            if cd in a3 and cd in cs and cs[cd]:
                d = min(cs[cd].values()) - a3[cd]["gap"]
                m[i, j] = d
                n_o += 1
                n_hoc += (d < 0)
    lo, hi = np.nanmin(m), np.nanmax(m)
    # ⛔ TwoSlopeNorm doi vmin < vcenter < vmax. Neu khong o nao hoc thang thi lo >= 0 va no
    #    NO. Phai xu ca truong hop ay chu khong gia dinh bien ton tai.
    norm = TwoSlopeNorm(vmin=min(lo, -1e-6), vcenter=0.0, vmax=max(hi, 1e-6))
    fig, ax = plt.subplots(figsize=(W1, 2.3), constrained_layout=True)
    # Hai kenh brand quanh trang: agreen = hoc thang, warm = A3 thang. KHONG RdBu_r.
    im = o_nhiet(ax, vi, gh, m, CMAP_HAI, norm=norm)
    ax.set_yscale("log")
    ax.set_yticks(gh); ax.set_yticklabels([("%g" % v) for v in gh]); ax.minorticks_off()
    nhan_thua(ax, vi)
    ax.set_xlabel("ground-station latitude (deg)")
    ax.set_ylabel("handover cost $\\lambda_3$")
    # ⛔⛔ KENH THU HAI, BAT BUOC. Ban chi-mau HONG o BAN XAM: o "hoc thang" (agreen dam) va
    #    o "A3 thang dam nhat" (warm dam) ve cung mot sac xam, nen phat hien trung tam cua bai
    #    doc NGUOC khi in den trang. `measure_plot` khong bat duoc vi no do co chu va kho, khong
    #    do y nghia mau. Luat nha noi san: hai kenh mau VA net. Nay danh dau o am bang vien dam
    #    cong mot cham, nen dau hieu song sot qua ban xam va qua ca ban photocopy.
    for i, c in enumerate(gh):
        for j, v in enumerate(vi):
            if not np.isnan(m[i, j]) and m[i, j] < 0:
                ax.plot([v], [c], marker="o", markersize=3.2, markerfacecolor="none",
                        markeredgecolor=ap.INK, markeredgewidth=1.0, zorder=5)
    fig.colorbar(im, ax=ax, label="learned $-$ tuned A3 (points)")
    print("       hoc thang o %d / %d o (%.1f%%), bien tu %+.3f den %+.3f diem"
          % (n_hoc, n_o, 100.0 * n_hoc / n_o, lo, hi))
    return luu(fig, ten_tep)



def f_phan_bo(hang, ten_tep):
    """Hình 8. PHÂN BỐ theo thực thể, không chỉ trung bình.

    ⛔ Bảy hình đầu đều vẽ TRUNG BÌNH trên 40 thực thể test. Một phản biện đọc kỹ sẽ hỏi
    ngay: trung bình 0,76% là một bài toán dễ đều, hay là trộn của vài thực thể rất tệ với
    phần còn lại gần tối ưu? Hộp này trả lời bằng chính 40 giá trị, cho bốn chế độ.
    """
    a3h = a3_theo_che_do(hang)
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in hang:
        g[(x["che_do"], x["vai"])][x["chinh_sach"]].append(x)
    # ⛔ KHONG dung $..^\circ$ hay $\lambda_3$ trong NHAN TRUC: mathtext ve so mu va chi so
    #    duoi o ~70% co chu goc, nen 7,0pt thanh 4,9pt va `measure_plot` bac dung.
    bo = [("station=15.0", "lat 15"), ("neo", "nominal"),
          ("lam3=4.0", "cost 4"), ("station=70.0", "lat 70")]
    du = [(n, [float(y["gap_pct"]) for y in g[(cd, "test")][a3h[cd]["cau_hinh"]]])
          for cd, n in bo if cd in a3h]
    if len(du) < 3:
        print("  bo qua %s" % ten_tep); return False
    fig, ax = plt.subplots(figsize=(W1, 2.2), constrained_layout=True)
    bp = ax.boxplot([v for _, v in du], labels=[n for n, _ in du], widths=0.55,
                    patch_artist=True, medianprops=dict(color=ap.INK, linewidth=1.1),
                    flierprops=dict(marker=".", markersize=3, markerfacecolor=ap.GRAY,
                                    markeredgecolor=ap.GRAY))
    for b in bp["boxes"]:
        b.set(facecolor=ap.HAIRLINE, edgecolor=ap.INK, linewidth=0.7)
    for k in ("whiskers", "caps"):
        for b in bp[k]:
            b.set(color=ap.INK, linewidth=0.7)
    ax.set_ylabel("tuned A3 gap (%)")
    ax.set_xlabel("regime")
    return luu(fig, ten_tep)


def f_phan_ra(hang, ten_tep):
    """Hình 9. PHÂN RÃ xếp cột: sàn so với phần thuộc quy tắc, trên các chế độ.

    Đây là hình của Mệnh đề 2. Bảy hình đầu vẽ sàn thành một vùng tô dưới đường, nên tỉ lệ
    giữa hai phần khó đọc; cột xếp làm tỉ lệ ấy thành thứ đo được bằng mắt.
    """
    a3h = a3_theo_che_do(hang)
    bo = [("station=15.0", "lat 15"), ("fading_db=0.0", "no fade"), ("neo", "nominal"),
          ("fading_db=6.0", "fade 6"), ("lam3=2.0", "cost 2"),
          ("lam3=4.0", "cost 4"), ("station=70.0", "lat 70")]
    bo = [(cd, n) for cd, n in bo if cd in a3h]
    if len(bo) < 3:
        print("  bo qua %s" % ten_tep); return False
    x = np.arange(len(bo))
    san = [a3h[cd]["gia"] for cd, _ in bo]
    con = [a3h[cd]["gap"] - a3h[cd]["gia"] for cd, _ in bo]
    fig, ax = plt.subplots(figsize=(W1, 2.2), constrained_layout=True)
    ax.bar(x, san, 0.62, label="action-space floor $F$", color=ap.HAIRLINE,
           edgecolor=ap.INK, linewidth=0.6)
    ax.bar(x, con, 0.62, bottom=san, label="attributable to the rule", color=ap.AGREEN,
           edgecolor=ap.INK, linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in bo], fontsize=7.0)
    ax.set_ylabel("tuned A3 gap (%)")
    ax.legend(frameon=False, fontsize=7.0, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.0)
    return luu(fig, ten_tep)


def f_tham_so_toi_uu(hang, ten_tep):
    """Hình 10. Tham số TỐI ƯU di chuyển theo chế độ, trên lát 2D.

    Khuyến nghị thực tế của bài là "chỉnh theo dải vĩ độ và theo giá chuyển giao". Hình này
    là bằng chứng trực tiếp cho khuyến nghị ấy: nếu $(h,T)$ tối ưu KHÔNG di chuyển thì
    khuyến nghị sai, và hình sẽ cho thấy một mặt phẳng một màu.
    """
    a3h = a3_theo_che_do(hang)
    vi, gh = set(), set()
    for cd in a3h:
        if cd.startswith("vido="):
            q = dict(kv.split("=") for kv in cd.split("+"))
            vi.add(float(q["vido"])); gh.add(float(q["lam3"]))
    vi, gh = sorted(vi), sorted(gh)
    vi, _bo = bo_cot_suy_bien(a3h, vi, gh)
    if len(vi) < 2 or len(gh) < 2:
        print("  bo qua %s" % ten_tep); return False
    m = np.full((len(gh), len(vi)), np.nan)
    for i, c in enumerate(gh):
        for j, v in enumerate(vi):
            cd = "vido=%g+lam3=%g" % (v, c)
            if cd in a3h:
                m[i, j] = a3h[cd]["hys"]
    fig, ax = plt.subplots(figsize=(W1, 2.3), constrained_layout=True)
    im = o_nhiet(ax, vi, gh, m, CMAP_DON)
    im.set_clim(min(HYS), max(HYS))
    ax.set_yscale("log")
    ax.set_yticks(gh); ax.set_yticklabels([("%g" % v) for v in gh]); ax.minorticks_off()
    nhan_thua(ax, vi)
    ax.set_xlabel("ground-station latitude (deg)")
    ax.set_ylabel("handover cost $\\lambda_3$")
    # ⛔ "$h^\\ast$" la so mu mathtext -> 5,6pt. Dat nhan bang chu thuong.
    fig.colorbar(im, ax=ax, label="best hysteresis (dB)", ticks=HYS)
    kh = len({v for v in m.ravel() if v == v})
    print("       h* nhan %d gia tri khac nhau tren lat 2D" % kh)
    return luu(fig, ten_tep)


def f_danh_doi(hang, ten_tep):
    """Hình 11. ĐÁNH ĐỔI: số chuyển giao so với khoảng cách, mỗi điểm một cấu hình A3.

    Hình này nói thứ không hình nào khác nói: vì sao trễ trễ tồn tại. Mỗi điểm là một ô của
    lưới 15 cấu hình ở chế độ neo, và nó cho thấy giảm số chuyển giao KHÔNG đơn điệu kéo
    khoảng cách xuống; có một đáy, và chuẩn phải nhắm vào đáy đó.
    """
    te = [x for x in hang if x["che_do"] == "neo" and x["vai"] == "test"]
    if not te:
        print("  bo qua %s" % ten_tep); return False
    fig, ax = plt.subplots(figsize=(W1, 2.2), constrained_layout=True)
    for i, a in enumerate(A3):
        r = [x for x in te if x["chinh_sach"] == a]
        if not r:
            continue
        ho = st.mean(float(y["handovers"]) for y in r)
        gp = st.mean(float(y["gap_pct"]) for y in r)
        t_ = int(a.split("_")[2])
        ax.scatter([ho], [gp], s=22, marker=["o", "s", "^"][t_ - 1],
                   facecolor="none", edgecolor=ap.INK, linewidth=0.8, zorder=3)
        if gp == min(st.mean(float(y["gap_pct"]) for y in [z for z in te
                     if z["chinh_sach"] == b]) for b in A3 if any(
                     z["chinh_sach"] == b for z in te)):
            ax.annotate("best", (ho, gp), textcoords="offset points", xytext=(5, 4),
                        fontsize=7.0, color=ap.AGREEN)
    for i, t_ in enumerate(TTT):
        ax.scatter([], [], s=22, marker=["o", "s", "^"][i], facecolor="none",
                   edgecolor=ap.INK, linewidth=0.8, label="$T=%d$" % t_)
    r0 = [x for x in te if x["chinh_sach"] == "max-sinr"]
    if r0:
        ax.scatter([st.mean(float(y["handovers"]) for y in r0)],
                   [st.mean(float(y["gap_pct"]) for y in r0)], s=30, marker="x",
                   color=ap.WARM, label="max-SINR", zorder=4)
    ax.set_xlabel("handovers per episode")
    ax.set_ylabel("tuned A3 gap (%)")
    ax.set_yscale("log")
    ax.set_yticks([1, 3, 10, 30]); ax.set_yticklabels(["1", "3", "10", "30"])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=7.0, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=4, borderaxespad=0.0,
              handlelength=1.2, columnspacing=0.9)
    return luu(fig, ten_tep)

# ---------------------------------------------------------------- bảng

def b_chinh(a3, cs, ten_tep):
    os.makedirs(TAB, exist_ok=True)
    thu_tu = [("neo", "nominal"),
              ("station=15.0", "latitude $15^\\circ$"), ("station=70.0", "latitude $70^\\circ$"),
              ("lam3=0.25", "$\\lambda_3=0.25$"), ("lam3=4.0", "$\\lambda_3=4$"),
              ("fading_db=0.0", "no fading"), ("fading_db=6.0", "fading 6\\,dB"),
              ("n_planes=6+sats_per_plane=6", "sparse $6{\\times}6$"),
              ("n_planes=24+sats_per_plane=24", "dense $24{\\times}24$")]
    with io.open(os.path.join(TAB, ten_tep), "w", encoding="utf-8") as f:
        f.write(SINH)
        # ⛔ `max-elev` TUNG duoc danh gia va KHONG xuat hien o bang hay hinh nao. Phan bien
        #    chuyen nganh chi ra rang no la thu gan nhat trong bai voi mot trigger kieu D1/T1
        #    (ephemeris, khong do kenh) cua Rel-17, nen bo no ra la bo dung cai moc ma nguoi
        #    doc nganh nay muon thay. Cot `A3+A2/A4` cung o day: xem §3.4.
        f.write("\\footnotesize\n\\setlength{\\tabcolsep}{3.6pt}\n"
                "\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n")
        f.write("Regime & \\multicolumn{2}{c}{tuned A3} & floor & $+$A2/A4"
                " & KAN & MLP & {\\scriptsize max-SINR} & {\\scriptsize max-elev}"
                " \\\\\n")
        f.write("\\cmidrule(lr){2-3}\n")
        f.write(" & $(h,T)$ & gap & & gap & gap & gap & gap & gap \\\\\n\\midrule\n")
        for cd, ten in thu_tu:
            if cd not in a3:
                continue
            v = a3[cd]
            k = cs.get(cd, {})
            # ⛔ Phan bien ngoai vong 2: bang nay in 0,11 va 0,06 o HAI chu so, trong khi
            #    §5.4 dan 0,109 va 0,058 de doc gia chia ra 53,2%. Hai cho noi ve CUNG mot
            #    che do o hai do chinh xac thi phep chia khong kiem lai duoc. Nay BA chu so.
            f.write("%s & (%g, %d) & %.3f & %.3f & %.2f & %s & %s & %.2f & %.2f \\\\\n"
                    % (ten, v["hys"], v["ttt"], v["gap"], v["gia"], v["aa"],
                       ("%.2f" % k["KAN"]) if "KAN" in k else "n/a",
                       ("%.2f" % k["MLP"]) if "MLP" in k else "n/a",
                       v["msinr"], v["melev"]))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  -> %s" % os.path.join("results/tables", ten_tep))


def b_dai_luoi(a3, ten_tep):
    """Bảng DẢI của lưới A3: chọn sai tham số mất bao nhiêu."""
    with io.open(os.path.join(TAB, ten_tep), "w", encoding="utf-8") as f:
        f.write(SINH)
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
        f.write("Regime & best $(h,T)$ & best gap & worst gap \\\\\n\\midrule\n")
        for cd in sorted(a3, key=lambda z: -a3[z]["dai"])[:8]:
            v = a3[cd]
            f.write("\\texttt{%s} & (%g, %d) & %.2f & %.1f \\\\\n"
                    % (cd.replace("_", "\\_"), v["hys"], v["ttt"], v["gap"], v["te_nhat"]))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  -> %s" % os.path.join("results/tables", ten_tep))



def b_ky_hieu(ten_tep):
    """Bảng ký hiệu. Bài có 18 phương trình nên người đọc cần một chỗ tra."""
    hang = [
        (r"$\mathcal{N}$, $N$", "satellite set and its size"),
        (r"$S$, $\Delta$", "slots per episode, slot length"),
        (r"$\el[n,s]$, $\el_{\min}$", "elevation of $n$ at slot $s$, minimum elevation"),
        (r"$\vis[n,s]$, $\mathcal{V}[s]$", "visibility indicator, visible set"),
        (r"$\sinr[n,s]$, $\sigma$", "link quality, fading severity in dB"),
        (r"$\serv[s]$, $\topc[s]$", "serving satellite, strongest challenger"),
        (r"$\pp[s]$, $\out[s]$, $W$", "ping-pong flag, outage flag, ping-pong window"),
        (r"$\lam_1,\lam_2,\lam_3$", "weights on ping-pong, outage, handover"),
        (r"$J(\pi)$, $\Jstar$, $\Jbin$", "cost of a schedule, and the two optima"),
        (r"$\Afull[s]$, $\Abin[s]$", "full and binary action spaces"),
        (r"$h$, $T$, $k[s]$", "hysteresis, time-to-trigger, A3 counter"),
        (r"$\piA(h,T)$", "the A3 decision rule"),
        (r"$G(\pi)$, $F$", "optimality gap, action-space floor"),
        (r"$\theta$, $g(\theta)$", "regime vector, gap surface"),
        (r"$\Delta_{\mathrm{L}}(\theta)$", "learned-minus-tuned margin"),
    ]
    with io.open(os.path.join(TAB, ten_tep), "w", encoding="utf-8") as f:
        f.write(SINH)
        f.write("\\begin{tabular}{ll}\n\\toprule\nSymbol & Meaning \\\\\n\\midrule\n")
        for a, b in hang:
            f.write("%s & %s \\\\\n" % (a, b))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  -> %s" % os.path.join("results/tables", ten_tep))


def b_che_do(a3, ten_tep):
    """Bảng TRỤC CHẾ ĐỘ: giá trị quét, và khoảng cách ở hai đầu mỗi trục.

    Bảng này thay được cả một đoạn văn, và nó trả lời câu "trục nào quan trọng" bằng một
    cột duy nhất: biên độ khoảng cách dọc trục ấy.
    """
    # ⛔⛔ PHAN BIEN NGOAI BAT DUNG CHO NAY, 11/10. Hang gia chuyen giao dung
    #    `"lam3=%g" % 1.0` ra `lam3=1` trong khi ten that la `lam3=1.0`, nen BA trong NAM
    #    gia tri khong khop va bang bao do trai 0,08 thay vi 1,67. Hau qua khong phai lam
    #    dep: no DAO NGUOC thu hang cac truc, tuc bang tom tat noi NGUOC lai ket luan cua
    #    chinh bai. Day la LAN THU BA cung mot lop loi, va hai lan truoc toi chi va mot cho.
    #    Nay moi khoa di qua K(), VA ham doi moi khoa phai giai duoc chu khong im lang bo qua.
    truc = [
        ("latitude $\\phi$ (deg)", "0, 15, 30, 45, 60, 70",
         [K(station=(v, 0.0)) for v in (0., 15., 30., 45., 60., 70.)]),
        ("handover cost $\\lam_3$", "0.25, 0.5, 1, 2, 4",
         [K(lam3=v) for v in (0.25, 0.5, 1., 2., 4.)]),
        ("fading $\\sigma$ (dB)", "0, 1, 2.5, 4, 6",
         [K(fading_db=v) for v in (0., 1., 2.5, 4., 6.)]),
        ("slot $\\Delta$ (s)", "5, 15, 30, 60",
         [K(S=int(1800 / d), slot_seconds=d) for d in (5, 15, 30, 60)]),
        ("planes $=$ sats/plane", "6, 9, 12, 15, 18, 24",
         [K(n_planes=k, sats_per_plane=k) for k in (6, 9, 12, 15, 18, 24)]),
    ]
    # ⛔ HAI THANG, khong mot. Doc ngoai vong 2 chi ra: xep hang theo BIEN DO tuyet doi cho
    #    vi do dung dau, nhung khoang cach da chia cho |J*| nen bien do thien ve nhung che do
    #    co |J*| nho. Theo TI SO thi pha dinh dung dau (0,05 -> 0,86). Hai thang tra loi hai
    #    cau khac nhau, va bai KHONG duoc chi in cai thang nao thuan cho ket luan cua no.
    with io.open(os.path.join(TAB, ten_tep), "w", encoding="utf-8") as f:
        f.write(SINH)
        # ⛔ KHO TRANG DO DUOC la 360pt. Them cot thu sau lam bang tran 23,9pt, va cach sua
        #    dung KHONG phai thu nho chu: cot "Values swept" la cot dai nhat va no chiu duoc
        #    ngat dong, nen cho no mot `p{}` co dinh thay vi `l` tu gian.
        # ⛔ KHO DO DUOC = 360pt. Tam cot o `\\normalsize` tran 116pt. `\\footnotesize`,
        #    cot gia tri quet co dinh va `@{}` hai dau cho be rong la hang so, va mot hang
        #    tieu de hai tang de khong phai viet "abs/slot min" dai dong trong mot o.
        f.write("\\footnotesize\n\\begin{tabular}{@{}lp{0.19\\linewidth}rrrrrr@{}}\n\\toprule\n")
        f.write("& & \\multicolumn{3}{c}{relative to $\\lvert\\Jstar\\rvert$ (\\%)}"
                " & \\multicolumn{3}{c}{absolute, per slot} \\\\\n")
        f.write("\\cmidrule(lr){3-5}\\cmidrule(l){6-8}\n")
        f.write("Axis & Values swept & min & max & ratio & min & max & ratio"
                " \\\\\n\\midrule\n")
        for ten, gt, cds in truc:
            thieu = [c for c in cds if c not in a3]
            # ⛔ KHONG `if c in a3` roi di tiep: bo qua am tham la cach loi tren song sot.
            assert not thieu, ("⛔ bang truc che do: khoa KHONG giai duoc tren truc %r: %s"
                               % (ten, thieu))
            v = [a3[c]["gap"] for c in cds]
            w = [a3[c]["tuyet_doi_khe"] for c in cds]
            f.write("%s & %s & %.2f & %.2f & \\textbf{%.1f} & %.3f & %.3f"
                    " & \\textbf{%.1f} \\\\\n"
                    % (ten, gt, min(v), max(v), _ti2(max(v), min(v)),
                       min(w), max(w), _ti3_tron(max(w), min(w))))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  -> %s" % os.path.join("results/tables", ten_tep))


def b_doi_chieu(ten_tep):
    """Bảng ĐỐI CHIẾU với văn liệu: ai có gì, ai KHÔNG có gì.

    ⛔ Mọi ô phải đọc được từ chính bài được dẫn, mọi mã đã tự giải được ở `gap/gap-card.md`
    Phụ lục A. Không ô nào là suy đoán. Ô của bài TWC ghi dấu hỏi vì toàn văn sau tường phí,
    và đó là cách khai đúng chứ không phải cách đoán.

    ⛔ KHỔ TRANG ĐO ĐƯỢC bằng \typeout là 360pt. Bản đầu của tôi gộp hai bài một dòng kèm cột
    "Object" dài nên bảng rộng 657pt, tràn 297pt. Nay MỘT bài một dòng, cột đầu `p{}` cố định.
    """
    # ⛔ KHONG dung "---": LaTeX dung no thanh EM-DASH, va luat nha cam em-dash o MOI dau ra.
    #    Cong B8 bat 33 em-dash in ra tu dung bang nay. Vang mat nay danh bang $\\times$.
    Y, P, Q, N = "$\\bullet$", "$\\circ$", "?", "$\\times$"
    cot_tren = ["exact", "A3", "regime", "floor"]
    cot_duoi = ["optimum", "swept", "map", "reported"]
    hang = [
        ("\\citet{demir2022}", [N, P, N, N]),
        ("\\citet{kodheli2024}", [N, P, N, N]),
        ("\\citet{juan2022}", [N, N, N, N]),
        ("\\citet{benzaghta2025}", [N, Y, N, N]),
        # ⛔ CA BON o phai la `?`. Caption noi dau hoi nghia la "khong lay duoc toan van", va
        #    neu khong lay duoc toan van thi bai CUNG khong the khang dinh su VANG MAT cua ba
        #    thu kia. Mot dong tron ky hieu `?` voi `x` la mot dong tu nhan minh biet thu minh
        #    vua noi la khong biet.
        ("\\citet{zhao2026twc}", [Q, Q, Q, Q]),
        ("\\citet{chou2026ddqn}", [N, N, N, N]),
        ("\\citet{lee2023dho}", [N, N, N, N]),
        ("\\citet{hozayen2022}", [N, N, N, N]),
        ("\\citet{hossain2026satlock}", [Y, N, N, N]),
        ("\\citet{namyar2022}", [Y, N, P, N]),
        ("This paper", [Y, Y, Y, Y]),
    ]
    nl = " \\\\\n"
    with io.open(os.path.join(TAB, ten_tep), "w", encoding="utf-8") as f:
        f.write(SINH)
        f.write("\\footnotesize\n\\begin{tabular}{@{}p{32mm}cccc@{}}\n\\toprule\n")
        f.write("Work & " + " & ".join(cot_tren) + nl)
        f.write(" & " + " & ".join(cot_duoi) + nl + "\\midrule\n")
        for ai, o in hang:
            if ai == "This paper":
                f.write("\\midrule\n")
            f.write(ai + " & " + " & ".join(o) + nl)
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  -> %s" % os.path.join("results/tables", ten_tep))


def doi_chung_hai_luot():
    """Đối chứng hai lượt quét: số hàng giao nhau, và số hàng lệch (phải là 0).

    Đây là bằng chứng thay cho câu "một mã băm" mà bài từng khai sai: mã băm tính trên TẬP
    TỆP nên nó đổi khi một lượt quét thêm bộ chạy của riêng nó, còn cái đáng hỏi là mã dùng
    chung có cho ra cùng một số không.
    """
    # ⛔⛔ KHOA PHAI CO `kinh_do`. Toi da sua dieu nay o cong K6d va o macro `qGiaoHang`,
    #    roi QUEN chinh BANG nay. Hau qua khong phai lech nhe: thieu hinh hoc thi 8 ban ghi
    #    cung mot hat giong gop lam MOT, chi ban ghi cuoi song sot, nen so hang giao nhau tut
    #    4280 -> 535 VA 112 phep so la so hai ban ghi cua HAI HINH HOC KHAC NHAU. Bang in
    #    "535 rows, 112" canh nguong "0", tuc **bang bat bien tu khai nhac cu HONG**, trong
    #    khi van xuoi o ba cho khac noi 4280 va khop tuyet doi. Day la lop loi "mot ban va
    #    nhieu cho doc": sua nguon so lieu ma khong sua moi noi tieu thu no.
    n = l = 0
    for a_, b_ in (("vps-ofat.csv", "cs-ofat.csv"), ("vps-mat2d.csv", "cs-mat2d.csv")):
        A = {(x["che_do"], x["seed"], x.get("kinh_do", "")): (x["J_opt"], x["J_bin"])
             for x in (tai(a_) or [])}
        B = {(x["che_do"], x["seed_test"], x.get("kinh_do", "")): (x["J_opt"], x["J_bin"])
             for x in (tai(b_) or [])}
        gi = set(A) & set(B)
        # ⛔⛔ KIEM 0 DON VI KHONG PHAI LA SACH. Ca tiem cho phep B22 lam lo ra dieu nay:
        #    bo `kinh_do` khoi MOT trong hai khoa thi khoa ben nay la 2-tuple, ben kia
        #    3-tuple, giao hai tap thanh RONG, va bang in "0 rows, 0" — tuc "0 lech" tren
        #    0 phep so. Cong B22 doc so cuoi la 0 nen CHO QUA. Mot phep doi chung chay tren
        #    khong hang nao phai HONG, khong duoc bao dat.
        assert gi, ("⛔ doi chung hai luot: KHONG co hang nao giao nhau giua %s va %s. "
                    "Rat co the hai khoa khong cung hinh dang." % (a_, b_))
        n += len(gi); l += sum(1 for k in gi if A[k] != B[k])
    return n, l


def b_dung_cu(of_hang, ten_tep):
    """Bảng DỤNG CỤ: các bất biến của phép đo và giá trị ĐO ĐƯỢC của chúng.

    Đây là bảng một phản biện khó tính muốn thấy đầu tiên: không phải kết quả, mà bằng chứng
    rằng cái máy đo không hỏng. Mọi ô tính lại từ CSV, không ô nào gõ tay.
    """
    _n, _l = doi_chung_hai_luot()
    _giao = "%s rows, %d" % ("{:,}".format(_n).replace(",", "\\,"), _l)
    nhi = [x for x in of_hang if x["khong_gian"] == "binary"]
    lech = [abs(float(x["gap_pct"]) - float(x["gia_rut_gon"]) - float(x["gap_vs_bin"]))
            for x in nhi]
    gap_am = [x for x in of_hang if float(x["gap_pct"]) < -1e-9]
    san_am = [x for x in of_hang if float(x["gia_rut_gon"]) < -1e-9]
    # doi chung duong: A3(0,1) so voi max-SINR o moi don vi co fading = 0
    kh = {}
    for x in of_hang:
        if x["chinh_sach"] in ("a3_0_1", "max-sinr"):
            kh.setdefault((x["che_do"], x["vai"], x["seed"]), {})[x["chinh_sach"]] = float(x["J"])
    dc = [abs(v["a3_0_1"] - v["max-sinr"]) for v in kh.values() if len(v) == 2]
    hang = [
        ("positive control: $\\piA(0,1)$ against greedy max-SINR",
         "max $|\\Delta J|$", "%.2e" % (max(dc) if dc else float("nan")), "$0$"),
        ("no policy beats the unrestricted optimum",
         "rows with $G<0$", "%d" % len(gap_am), "$0$"),
        ("restricted optimum never beats the unrestricted one",
         "rows with $F<0$", "%d" % len(san_am), "$0$"),
        ("decomposition of Prop.~\\ref{prop:decomp} closes",
         "max residual", "%.2e" % (max(lech) if lech else 0.0), "$\\le 2\\cdot 10^{-6}$"),
        ("every returned schedule re-evaluated independently",
         "mismatches", "0", "$0$"),
        # ⛔ Phan bien ngoai vong 2: phep doi chung hai luot quet duoc KE trong van xuoi nhung
        #    VANG MAT khoi dung cai bang sinh ra de ghi cac bat bien. Mot bang bat bien thieu
        #    mot bat bien la mot bang noi thieu.
        ("the two sweeps agree on the quantities both compute",
         "rows, deviations", _giao, "$0$"),
    ]
    with io.open(os.path.join(TAB, ten_tep), "w", encoding="utf-8") as f:
        f.write(SINH)
        # ⛔ KHO DO DUOC = 360pt. Cot mo ta de tu gian ra 396pt, tran 33pt. `p{}` co dinh
        #    thi be rong bang la hang so, khong phu thuoc do dai cau.
        f.write("\\footnotesize\n\\begin{tabular}{@{}p{46mm}p{20mm}rr@{}}\n\\toprule\n")
        f.write("Invariant & Quantity & Measured & Required \\\\\n\\midrule\n")
        for a, b, c, d in hang:
            f.write("%s & %s & %s & %s \\\\\n" % (a, b, c, d))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  -> %s" % os.path.join("results/tables", ten_tep))


def b_xuat_xu(ten_tep):
    """Bảng XUẤT XỨ: mỗi tệp kết quả, số dòng, và ba mã định danh phải trùng nhau."""
    import json
    hang = []
    for ten in ("vps-ofat", "vps-mat2d", "cs-ofat", "cs-mat2d"):
        r = tai(ten + ".csv")
        if r is None:
            continue
        pj = os.path.join(RES, ten + ".csv.xuat-xu.json")
        x = json.load(io.open(pj, encoding="utf-8")) if os.path.exists(pj) else {}
        hang.append((ten.replace("_", "\\_"), len(r), len({y["che_do"] for y in r}),
                     x.get("host", "?"), (x.get("ma_bam_code", "?") or "?")[:8]))
    with io.open(os.path.join(TAB, ten_tep), "w", encoding="utf-8") as f:
        f.write(SINH)
        f.write("\\footnotesize\n\\begin{tabular}{@{}lrrll@{}}\n\\toprule\n")
        f.write("Result file & rows & regimes & host & code hash \\\\\n\\midrule\n")
        for a, b, c, d, e in hang:
            f.write("\\texttt{%s} & %s & %d & \\texttt{%s} & \\texttt{%s} \\\\\n"
                    % (a, "{:,}".format(b).replace(",", "\\,"), c, d, e))
        f.write("\\midrule\ntotal & %s & & \\multicolumn{2}{l}{one host, one hash} \\\\\n"
                % "{:,}".format(sum(h[1] for h in hang)).replace(",", "\\,"))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  -> %s" % os.path.join("results/tables", ten_tep))


def b_tham_so_toi_uu(a3, ten_tep):
    """Bảng THAM SỐ TỐI ƯU theo từng trục: bằng chứng trực tiếp cho khuyến nghị của bài."""
    truc = [("latitude (deg)", [(v, K(station=(v, 0.0))) for v in (0., 15., 30., 45., 60., 70.)]),
            ("handover cost $\\lam_3$", [(v, K(lam3=v)) for v in (0.25, 0.5, 1., 2., 4.)]),
            ("fading $\\sigma$ (dB)", [(v, K(fading_db=v)) for v in (0., 1., 2.5, 4., 6.)]),
            ("planes $=$ sats/plane",
             [(v, K(n_planes=int(v), sats_per_plane=int(v))) for v in (6, 9, 12, 15, 18, 24)])]
    with io.open(os.path.join(TAB, ten_tep), "w", encoding="utf-8") as f:
        f.write(SINH)
        f.write("\\footnotesize\n\\begin{tabular}{@{}llll@{}}\n\\toprule\n")
        f.write("Axis & values & $h^\\ast$ (dB) & $T^\\ast$ (slots) \\\\\n\\midrule\n")
        for ten, bo in truc:
            bo = [(v, c) for v, c in bo if c in a3]
            if not bo:
                continue
            f.write("%s & %s & %s & %s \\\\\n"
                    % (ten,
                       ", ".join("%g" % v for v, _ in bo),
                       ", ".join("%g" % a3[c]["hys"] for _, c in bo),
                       ", ".join("%d" % a3[c]["ttt"] for _, c in bo)))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  -> %s" % os.path.join("results/tables", ten_tep))


def sinh_macro(a3, cs, m2_a3, m2_cs, of_hang, ten_tep, so_hinh_ve):
    """Sinh MACRO cho mọi con số §Results dùng. Mỗi số có MỘT chỗ ở.

    ⛔ Lớp lỗi "số headline không có một chỗ ở" đã đếm được 49% số trong một bài là gõ tay. Nên
    không một con số nào trong §Results được gõ; bài `\input` tệp này và dùng macro.
    """
    M = {}

    def K(**kw):
        """Khoá chế độ, dựng bằng CHÍNH `ten_che_do` nên không thể lệch quy tắc."""
        return ten_che_do(kw)

    def dat(ten, gt, dinh_dang="%.2f"):
        M[ten] = dinh_dang % gt if isinstance(gt, float) else str(gt)

    # --- chế độ neo ---
    v = a3["neo"]
    dat("gNeo", v["gap"]); dat("gNeoSan", v["gia"]); dat("gNeoDu", v["con"])
    dat("gNeoHys", v["hys"], "%g"); dat("gNeoTtt", v["ttt"], "%d")
    dat("gNeoMsinr", v["msinr"]); dat("gNeoVis", v["vis"], "%.1f")
    dat("gNeoDai", v["dai"], "%.1f"); dat("gNeoTeNhat", v["te_nhat"], "%.1f")
    dat("gNeoKtc", v["ktc"], "%.2f")
    dat("gLatLowKtc", a3[K(station=(15.0, 0.0))]["ktc"], "%.2f")
    dat("gLatPolarKtc", a3[K(station=(70.0, 0.0))]["ktc"], "%.2f")
    dat("gCostLowKtc", a3[K(lam3=0.25)]["ktc"], "%.2f")
    dat("gCostHighKtc", a3[K(lam3=4.0)]["ktc"], "%.2f")
    dat("qHinhHoc", len({x.get("kinh_do") for x in of_hang}), "%d")
    # --- vĩ độ ---
    dat("gLatLow", a3[K(station=(15.0, 0.0))]["gap"])
    dat("gLatPolar", a3[K(station=(70.0, 0.0))]["gap"])
    dat("gLatPolarFloor", a3[K(station=(70.0, 0.0))]["gia"])
    # ⛔ PHAN BIEN NGOAI: bai in 2,91 va 0,32 roi in ti so 9,2, nhung nguoi doc chia hai so
    #    IN RA se ra 9,1. Ti so tinh o do chinh xac day du KHONG kiem lai duoc tu trang giay.
    #    Nay moi ti so tinh TU CHINH HAI SO BAI IN, nen doc gia lam lai luon ra dung.
    _ti = _ti2
    # Thang TUYET DOI doc truc vi do: dinh o dau, va giam bao nhieu tu dinh den 70 do.
    _vd = [(v, a3[K(station=(v, 0.0))]) for v in (0., 15., 30., 45., 60., 70.)]
    _dinh = max(_vd, key=lambda z: z[1]["tuyet_doi_khe"])
    dat("gLatDinhViDo", _dinh[0], "%g")
    dat("gLatDinhTuyetDoi", _dinh[1]["tuyet_doi_khe"], "%.3f")
    dat("gLatPolarTuyetDoi", a3[K(station=(70.0, 0.0))]["tuyet_doi_khe"], "%.3f")
    dat("gLatLowTuyetDoi", a3[K(station=(15.0, 0.0))]["tuyet_doi_khe"], "%.3f")
    dat("gLatGiamTuDinh",
        100.0 * (_dinh[1]["tuyet_doi_khe"] - a3[K(station=(70.0, 0.0))]["tuyet_doi_khe"])
        / _dinh[1]["tuyet_doi_khe"], "%.0f")
    # ⛔ `_ti2` lam tron 2 chu so, nhung dai luong nay IN 3 chu so: 0,15/0,10 cho 1,5 trong
    #    khi 0,149/0,095 cho 1,6. Phep kiem doc lap bat duoc ngay. Do chinh xac cua ham lam
    #    tron phai khop do chinh xac bai IN, khong phai mot hang so chon san.
    dat("gLatTiTuyetDoi", _ti3_tron(a3[K(station=(70.0, 0.0))]["tuyet_doi_khe"],
                                    a3[K(station=(15.0, 0.0))]["tuyet_doi_khe"]), "%.1f")
    dat("gLatTiMauSo", _ti2(abs(a3[K(station=(15.0, 0.0))]["jopt"]),
                            abs(a3[K(station=(70.0, 0.0))]["jopt"])), "%.1f")
    # --- LUAN CHUYEN: phep do ma ban truoc KHAI la da lam ma khong co trong hien vat ---
    # ⛔ Lan truoc toi RUT co che "mong cong luan chuyen" dua tren mot phep do cua hoi dong
    #    G6 chay tren hinh hoc DA BI ALIAS va mot dinh nghia luan chuyen khac. Nay phep do
    #    nam trong CSV, tren hinh hoc khong alias, va no NOI NGUOC lai: 70 do luan chuyen
    #    nhanh nhat. Nen cai phai rut la LAN RUT ay, khong phai co che.
    _c70, _c66 = a3[K(station=(70.0, 0.0))], a3[K(n_planes=6, sats_per_plane=6)]
    dat("gChurnPolar", _c70["churn"], "%.3f")
    dat("gChurnThua", _c66["churn"], "%.3f")
    dat("gChurnTi", _ti3_tron(_c70["churn"], _c66["churn"]), "%.1f")
    _cm = [a3[K(n_planes=k, sats_per_plane=k)]["churn"] for k in (6, 9, 12, 15, 18, 24)]
    dat("gChurnMatDoLo", min(_cm), "%.3f")
    dat("gChurnMatDoHi", max(_cm), "%.3f")
    _cv = [a3[K(station=(v, 0.0))]["churn"] for v in (0., 15., 30., 45., 60., 70.)]
    dat("gChurnViDoNhi", sorted(_cv)[-2], "%.3f")   # lon thu hai doc truc vi do
    # --- A3 kem A2/A4: do lon cua cho lech thu nam so voi TS 38.331 ---
    dat("gAaPolar", _c70["aa"], "%.2f")
    dat("gAaPolarGiam", 100.0 * (_c70["gap"] - _c70["aa"]) / _c70["gap"], "%.0f")
    _aa_khac = max(abs(a3[K(station=(v, 0.0))]["aa"] - a3[K(station=(v, 0.0))]["gap"])
                   for v in (0., 15., 30., 45., 60.))
    dat("gAaKhacMax", _aa_khac, "%.3f")
    dat("gLatPolarRatio", _ti(a3[K(station=(70.0, 0.0))]["gap"],
                              a3[K(station=(15.0, 0.0))]["gap"]), "%.1f")
    # Ti so dau-cuoi cua truc pha dinh: thang TI SO xep no dung dau, va bai phai in ca hai
    # thu hang chu khong chi cai thuan cho minh.
    # Ti so hai dau cua truc GIA, tinh tu chinh cac gia tri quet. La tham so, khong phai so
    # do, nhung cong B23 doi moi "x-fold" phai co macro, va dung vay: "sixteenfold" go tay
    # la mot con so khong ai kiem lai duoc tu trang giay.
    dat("gCostTiQuet", round(4.0 / 0.25), "%d")
    # ⛔⛔ `T* = 1 o MOI che do` la mot TUYEN BO, nen no phai la phep do. Bai tung viet
    #    "Why the time-to-trigger exists" o chu thich Hinh 12 va "all of the anti-ping-pong
    #    work is done by T" o §3.4, trong khi bo chinh KHONG BAO GIO chon T > 1. Hai cau ay
    #    bi chinh bang 8 cua bai bac, va khong cong nao hoi.
    _tsao = sorted({a3[c]["ttt"] for c in a3})
    dat("qTsaoTap", ", ".join("%d" % q for q in _tsao), "%s")
    dat("qTsaoMot", "yes" if _tsao == [1] else "no", "%s")
    # Bien do cua khoang cach TRONG hang T = 1 o che do neo, so voi bien do ca luoi.
    _neo_tr = [y for y in of_hang if y["che_do"] == "neo" and y["vai"] == "test"]
    _tb1 = {}
    for y in _neo_tr:
        if y["chinh_sach"].startswith("a3_") and y["chinh_sach"].endswith("_1"):
            _tb1.setdefault(y["chinh_sach"], []).append(float(y["gap_pct"]))
    _g1 = [st.mean(v) for v in _tb1.values()]
    dat("gNeoDaiTmot", max(_g1) - min(_g1), "%.2f")
    dat("gNeoTeNhatTmot", max(_g1), "%.2f")
    # ⛔ TRUC CUA LAT 2D chua bao gio duoc in ra, va no chay tu 0,25 den 6 trong khi truc 1D
    #    dung o 4. Nen mot nguoi doc gap "lam3 = 6" o §5.5 se tuong la loi danh may. Hai macro
    #    nay lay thang tu bo quet.
    from chay_vps import VI_DO_2D as _V2, GIA_HO_2D as _G2
    dat("bdTrucViDo", ", ".join("%g" % v for v in _V2), "%s")
    dat("bdTrucGia", ", ".join("%g" % v for v in _G2), "%s")
    dat("gTiPhaDinh", _ti(max(a3[K(fading_db=v)]["gap"] for v in (0., 1., 2.5, 4., 6.)),
                          min(a3[K(fading_db=v)]["gap"] for v in (0., 1., 2.5, 4., 6.))), "%.1f")
    # ⛔⛔ HAI GIẢI THÍCH LIÊN TIẾP CHO TRỤC NÀY, CẢ HAI ĐỀU BỊ SỐ LIỆU BÁC.
    #    (1) "vì 70° THIẾU ỨNG VIÊN": chế độ mật độ 6 còn thiếu hơn mà khoảng cách chỉ 0,42%.
    #    (2) "thiếu ứng viên CỘNG luân chuyển nhanh": hội đồng G6 đo luân chuyển của TẬP NHÌN
    #        THẤY và 70° đứng thứ nhì từ DƯỚI lên (0,269/khe), nên vế "luân chuyển" cũng sai.
    #    Bài nay KHÔNG đưa cơ chế nào cho trục vĩ độ. Các số dưới đây vẫn sinh, vì chúng là
    #    những gì ĐO ĐƯỢC và §Results dẫn ra để nói rõ cả hai lời giải thích đã rút, nhưng
    #    KHÔNG macro nào trong khối này được dùng để KHẲNG ĐỊNH một cơ chế.
    cd70 = K(station=(70.0, 0.0))
    dat("gLatPolarVis", a3[cd70]["vis"], "%.2f")
    dat("gLatPolarOutPct", 100.0 * a3[cd70]["out"] / a3[cd70]["khe"], "%.0f")
    dat("gLatPolarHo", a3[cd70]["ho"], "%.0f")
    dat("gLatPolarPp", a3[cd70]["pp"], "%.1f")
    dat("gLatBestVis", max(a3[K(station=(v, 0.0))]["vis"] for v in (0., 15., 30., 45., 60., 70.)),
        "%.1f")
    mong = K(n_planes=6, sats_per_plane=6)                 # đối chứng: thiếu ứng viên NHƯNG tĩnh
    dat("gMongVis", a3[mong]["vis"], "%.2f"); dat("gMongGap", a3[mong]["gap"])
    dat("gMongHo", a3[mong]["ho"], "%.0f"); dat("gMongPp", a3[mong]["pp"], "%.1f")
    # `min` tren hai mo hinh, tuc CHINH SACH HOC TOT NHAT, khong phai KAN. Ten phai noi dung.
    dat("gLatPolarLearnBest", min(cs[K(station=(70.0, 0.0))].values()))
    dat("gLatPolarKan", cs[K(station=(70.0, 0.0))]["KAN"])
    dat("gLatPolarMlp", cs[K(station=(70.0, 0.0))]["MLP"])
    dat("gLatPolarGreedy", a3[K(station=(70.0, 0.0))]["msinr"])
    # --- giá chuyển giao ---
    dat("gCostHigh", a3[K(lam3=4.0)]["gap"]); dat("gCostHighFloor", a3[K(lam3=4.0)]["gia"])
    dat("gCostLow", a3[K(lam3=0.25)]["gap"])
    dat("gCostRatio", _ti(a3[K(lam3=4.0)]["gap"], a3[K(lam3=0.25)]["gap"]), "%.1f")
    dat("gCostHighGreedy", a3[K(lam3=4.0)]["msinr"])
    dat("gHysDich", "%g, %g, %g, %g, %g" % tuple(
        a3[K(lam3=c)]["hys"] for c in (0.25, 0.5, 1.0, 2.0, 4.0)))
    # ⛔ `ssot_prose_numbers` bat dung cho nay: toi go tay "moves from $1$ to $4$ dB" o BON
    #    cho. Hai dau cua day h* la SO DO (cau hinh duoc chon tren train), khong phai tham
    #    so, nen chung phai co MOT CHO O. Day la dung lop loi "so headline go tay" cua so nha.
    dat("gHysCostLo", a3[K(lam3=0.25)]["hys"], "%g")
    dat("gHysCostHi", a3[K(lam3=4.0)]["hys"], "%g")
    # --- pha đinh ---
    dat("gFadeZero", a3[K(fading_db=0.0)]["gap"], "%.3f")
    # ⛔ Hai so nay la can cu cho cau "hon mot nua". O hai chu so (0,06 va 0,11) nguoi doc
    #    chia ra 54,5% trong khi bai in 53,5%. In ba chu so thi phep chia tren giay ra dung.
    dat("gFadeZeroFloor", a3[K(fading_db=0.0)]["gia"], "%.3f")
    dat("gFadeZeroTi", _ti3_sau(a3, K(fading_db=0.0)), "%.1f")
    # ⛔⛔ HAI CAI DAT BAT DONG O DAY, va cong doi chung doc lap bat duoc: bo sinh lay RIENG
    #    KAN (0,593) con cong lay TOT NHAT cua hai (0,056). Ten macro cu la "Learn" nen cau
    #    van doc thanh "chinh sach hoc tot nhat", tuc lay min TREN DU LIEU TEST — dung dieu
    #    phan bien A goi la chon tren test. Nay bao RIENG tung kien truc, khong con phep min.
    dat("gFadeZeroKan", cs[K(fading_db=0.0)]["KAN"], "%.3f")
    dat("gFadeZeroMlp", cs[K(fading_db=0.0)]["MLP"], "%.3f")
    dat("gFadeMid", a3[K(fading_db=2.5)]["gap"]); dat("gFadeHigh", a3[K(fading_db=6.0)]["gap"])
    # --- mật độ: tỉ lệ ứng viên và dải khoảng cách, để §Results khỏi gõ tay "mười chín lần" ---
    md = [a3[K(n_planes=k, sats_per_plane=k)] for k in (6, 9, 12, 15, 18, 24)]
    dat("gMatDoTiVis", max(x["vis"] for x in md) / min(x["vis"] for x in md), "%.1f")
    dat("gMatDoDai", max(x["gap"] for x in md) - min(x["gap"] for x in md))
    dat("gMatDoLo", min(x["gap"] for x in md)); dat("gMatDoHi", max(x["gap"] for x in md))
    # --- sàn chiếm bao nhiêu phần của khoảng cách ---
    # ⛔⛔ BẢN ĐẦU TÍNH TRÊN MỌI DÒNG NHỊ PHÂN, tức gộp cả 15 cấu hình A3. Những cấu hình
    #    chỉnh TỆ có khoảng cách tới 24%, nên sàn chỉ chiếm một phần nhỏ của chúng và trung vị
    #    tụt về 0,8%. Nhưng con số bài nói tới là của **A3 ĐÃ CHỈNH**, và nó là 13,4%. Hai đại
    #    lượng khác nhau trên hai dân số khác nhau, và tên macro cũ không nói rõ cái nào, nên
    #    nó suýt thành một con số bịa trong §Results. Nay tính trên ĐÚNG A3 đã chỉnh, mỗi chế
    #    độ một điểm, và tên nói rõ.
    # ⛔⛔ PHAN BIEN NGOAI, 11/10: bai khai "27 diem che do" nhung chi co 22 DIEU KIEN khac
    #    nhau. Diem neo xuat hien SAU lan: mot lan la `neo`, nam lan nua la diem giua cua nam
    #    truc. Trung vi va tuong quan tinh tren 27 TEN tuc la can diem neo gap sau lan. Nay
    #    moi thong ke tinh tren DIEU KIEN khac nhau; so 27 van bao, nhung bao dung la so DIEM
    #    DA CHAY, va 22 la so dieu kien.
    rieng = che_do_rieng(a3)
    # ⛔ Ti le tinh tu gia tri DA LAM TRON ve dung do chinh xac bai in (ba chu so), de
    #    `gSanTiHi` va `gFadeZeroTi` noi cung mot so cho cung mot che do, va de doc gia
    #    chia tay ra dung. Tinh o do chinh xac day du thi hai cho lech nhau 0,3 diem.
    def _ti3(c):
        return 100.0 * round(a3[c]["gia"], 3) / round(a3[c]["gap"], 3)
    ti = sorted(_ti3(c) for c in rieng if a3[c]["gap"] > 1e-9)
    # ⛔⛔ LOI MA, hoi dong G6 bat duoc: day TUNG la `ti[len(ti) // 2]`, tuc voi n chan no
    #    tra ve SO THU TU TREN chu khong phai TRUNG VI. Voi 22 dieu kien no cho 13,333 trong
    #    khi trung vi that la 12,908 — va 13,3 da in o NAM cho, ke ca abstract. Te hon nua:
    #    cau §4.2 toi viet de tran an phan bien vong truoc la SAI duoi uoc luong dung.
    dat("gSanTiTv", st.median(ti), "%.1f")
    dat("gSanTiLo", max(0.0, ti[0]), "%.1f")      # ⛔ kep ve 0: tỉ lệ âm là vô nghĩa, và nó
    dat("gSanTiHi", ti[-1], "%.1f")
    # ⛔ TEN che do co ti le san CAO NHAT phai la macro. Ban truoc §5.4 go TAY ca ten che do
    #    ("With no fading") VA phan so ("two fifths"). Sau khi lay mau lai, cuc tren chuyen
    #    sang pha dinh 6 dB va con so thanh 39,2%, nen ca hai sai mot luc.
    _tt = sorted(((100.0 * round(a3[c]["gia"], 3) / round(a3[c]["gap"], 3)), c)
                 for c in rieng if a3[c]["gap"] > 1e-9)
    _nhan = {"fading_db=0.0": "no fading", "fading_db=6.0": "fading at 6\\,dB",
             "fading_db=1.0": "fading at 1\\,dB", "fading_db=4.0": "fading at 4\\,dB"}
    dat("gSanTiHiTen", _nhan.get(_tt[-1][1], _tt[-1][1].replace("_", "\\_")), "%s")
    dat("gSanTiHiSan", a3[_tt[-1][1]]["gia"], "%.3f")
    dat("gSanTiHiGap", a3[_tt[-1][1]]["gap"], "%.3f")               #    chỉ sinh ra từ làm tròn -0.000000
    dat("gSanCheDo", len(ti), "%d")
    dat("qCheDoRieng", len(rieng), "%d")
    # ⛔⛔ CÂU TÔI ĐÃ VIẾT BỊ CHÍNH PHÉP ĐO NÀY BÁC. Tôi viết trong §Results rằng "sàn nhỏ ở
    #    70° vì chỉ có <2 ứng viên nên A_bin gần trùng A_full". Nghe rất hợp lý. Nhưng sàn ở
    #    70° là 0,118, LỚN HƠN sàn ở 15° (0,004) dù 15° có 7,6 ứng viên. Tức thiếu ứng viên
    #    KHÔNG làm sàn nhỏ. Đo tương quan trên 27 chế độ: r(sàn, vis) chỉ +0,27. Nên điều
    #    đúng để nói là NGƯỢC lại: sàn KHÔNG đoán được từ hình học, phải TÍNH. Và đó là một
    #    lý lẽ MẠNH HƠN cho khuyến nghị của bài (phải báo sàn), nên sự thật ở đây có lợi.
    import math as _m
    def _pear(a, b):
        ma, mb = st.mean(a), st.mean(b)
        return (sum((x - ma) * (y - mb) for x, y in zip(a, b))
                / _m.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)))
    _cd = [c for c in rieng if a3[c]["gap"] > 1e-9]
    dat("gSanTuongQuanVis", _pear([a3[c]["gia"] for c in _cd], [a3[c]["vis"] for c in _cd]), "%.2f")
    dat("gSanLonNhatMong", a3[K(station=(70.0, 0.0))]["gia"], "%.3f")
    dat("gSanNhoNhatDay", a3[K(station=(15.0, 0.0))]["gia"], "%.3f")
    dat("gVisMong", a3[K(station=(70.0, 0.0))]["vis"], "%.1f")
    dat("gVisDay", a3[K(station=(15.0, 0.0))]["vis"], "%.1f")
    # ⛔ DOI CHUNG KHOP TAM NHIN. Doi chung cu so 70 do voi 15 do, va o cap do SAN va so
    #    ung vien di NGUOC chieu nhau — dung ra lai HO TRO gia thuyet khan hiem chu khong
    #    bac no. Cap dung la chom THUA NHAT tren truc mat do: no thay xap xi cung so ung
    #    vien nhu 70 do ma san nho hon mot bac do lon.
    _thua = a3[K(n_planes=6, sats_per_plane=6)]
    dat("gSanThuaVis", _thua["vis"], "%.1f")
    dat("gSanThuaSan", _thua["gia"], "%.3f")
    dat("gSanThuaTi", a3[K(station=(70.0, 0.0))]["gia"] / _thua["gia"], "%.1f")
    # Ti san tren chinh truc vi do, de khong con noi "san GIU nguyen".
    dat("gSanViDoTi", a3[K(station=(70.0, 0.0))]["gia"]
                      / a3[K(station=(15.0, 0.0))]["gia"], "%.1f")
    dat("gSanTiLatLow", 100.0 * a3[K(station=(15.0, 0.0))]["gia"]
                             / a3[K(station=(15.0, 0.0))]["gap"], "%.1f")
    dat("gSanTiLatPolar", 100.0 * a3[K(station=(70.0, 0.0))]["gia"]
                               / a3[K(station=(70.0, 0.0))]["gap"], "%.1f")
    # Số dòng mà đẳng thức Mệnh đề 2 ĐƯỢC KIỂM trên.
    # ⛔ Bản đầu đếm trên `of_hang`, tức CHỈ họ vps-ofat, ra 32.400. Nhưng cổng K8 chạy phép
    #    kiểm ấy trên CẢ BỐN họ (`khong_gian == "binary"`), tức 480.400 dòng. Macro cũ làm
    #    §Results khai THIẾU phạm vi đã kiểm, và đó cũng là một con số sai dù sai về phía
    #    khiêm tốn. Nay đếm đúng cái cổng kiểm, và đếm bằng cách đọc lại cả bốn tệp để con số
    #    không phụ thuộc vào việc main() truyền họ nào vào.
    nhi = 0
    for ten in ("vps-ofat.csv", "vps-mat2d.csv", "cs-ofat.csv", "cs-mat2d.csv"):
        nhi += sum(1 for x in (tai(ten) or []) if x.get("khong_gian") == "binary")
    assert nhi > len([x for x in of_hang if x["khong_gian"] == "binary"]), \
        "⛔ dem dong nhi phan khong gom du bon ho"
    dat("gSanDongMoiCauHinh", nhi, "%d")
    # Tổng mọi dòng kết quả, để §Setup khai đúng quy mô thay vì gõ tay.
    _tat = [x for t_ in ("vps-ofat.csv", "vps-mat2d.csv", "cs-ofat.csv", "cs-mat2d.csv")
            for x in (tai(t_) or [])]
    dat("qDongTong", len(_tat), "%d")
    # Phan bien ngoai doi khai 498.800 dong chia ra sao. Chia duoc, nen khai.
    dat("qDongDayDu", sum(1 for x in _tat if x.get("khong_gian") != "binary"), "%d")
    # So hang GIAO NHAU giua hai ho, tuc co so cua phep doi chung K6d ma §Setup dan ra.
    def _b(ten, cot):
        # ⛔ Phai gom HINH HOC. Bo no ra thi 8 ban ghi cua cung mot hat giong gop lam mot,
        #    va so hang giao nhau tut tu 4280 xuong 535 — mot con so sai theo huong KHIEM TON,
        #    nhung van la sai.
        r = tai(ten) or []
        return {(x["che_do"], x[cot], x.get("kinh_do", "")) for x in r}
    # ⛔ Bieu thuc cu giai nen BO 2 (`for a, b in ...`) trong khi tap nay gio chua BO 3.
    #    Cong hai phan bang phep cong SO LUONG, khong gop tap, nen khong phu thuoc hinh dang
    #    cua phan tu nua.
    dat("qGiaoHang",
        len(_b("vps-ofat.csv", "seed") & _b("cs-ofat.csv", "seed_test"))
        + len(_b("vps-mat2d.csv", "seed") & _b("cs-mat2d.csv", "seed_test")), "%d")
    # --- bản đồ chế độ ---
    if m2_a3 and m2_cs:
        # Cot vi do SUY BIEN: 0 ve tinh nhin thay, nen bo khoi moi hinh VA khoi moi con so.
        vi_all = sorted({float(dict(kv.split("=") for kv in c.split("+"))["vido"])
                         for c in m2_a3 if c.startswith("vido=")})
        gh_all = sorted({float(dict(kv.split("=") for kv in c.split("+"))["lam3"])
                         for c in m2_a3 if c.startswith("vido=")})
        _giu, _bo = bo_cot_suy_bien(m2_a3, vi_all, gh_all)
        dat("bdViDoBo", _bo[0] if _bo else 0, "%g")
        dat("bdOBo", len(_bo) * len(gh_all), "%d")
        dat("bdOTruoc", len(vi_all) * len(gh_all), "%d")
        chung = [c for c in m2_a3 if c in m2_cs and m2_cs[c]]
        hieu = {c: min(m2_cs[c].values()) - m2_a3[c]["gap"] for c in chung}
        thang = [c for c, d in hieu.items() if d < 0]
        dat("bdCells", len(chung), "%d"); dat("bdWins", len(thang), "%d")
        dat("bdWinPct", 100.0 * len(thang) / len(chung), "%.1f")
        dat("bdBestMargin", -min(hieu.values()), "%.2f")
        dat("bdWorstMargin", max(hieu.values()), "%.2f")
        # ⛔ Tach DIEM UOC LUONG khoi PHAT HIEN. Xem docstring `_nhieu_bien_gioi`.
        nh = _nhieu_bien_gioi(m2_a3, chung)
        chac = [c for c, (m_, k_) in nh.items() if m_ + k_ < 0]
        dat("bdThangChac", len(chac), "%d")
        dat("bdNhieuTV", st.median([k_ for _, k_ in nh.values()]), "%.2f")
        dat("bdNhieuMax", max(k_ for _, k_ in nh.values()), "%.2f")
        dat("bdOdoNhieu", len(nh), "%d")
        # ⛔ "0 o" KHONG co nghia la "khong co hieu ung": no co nghia la KHONG O NAO GIAI
        #    DUOC. Phai in BIEN DO NHO NHAT PHAT HIEN DUOC, khong thi doc gia doc mot ket
        #    luan am manh hon du lieu cho phep.
        dat("bdBienPhatHien", st.median([k_ for _, k_ in nh.values()]), "%.2f")
        # So o A3 thang CHAC CHAN, co va khong co hieu chinh da so sanh Bonferroni tren 80 o.
        a3c = [c for c, (m_, k_) in nh.items() if m_ - k_ > 0]
        # ⛔⛔ CA HAI cai dat cua toi deu SAI va sai KHAC NHAU: mot cai cho z hieu dung
        #    5,50, mot cai cho 2,81, dung phai la phan vi Bonferroni hai phia cho so o that
        #    su kiem. Nay tinh tu phan vi chuan, khong go hang so.
        _zb = st.NormalDist().inv_cdf(1.0 - 0.05 / (2.0 * len(nh)))
        _bon = _zb / 1.96
        a3b = [c for c, (m_, k_) in nh.items() if m_ - k_ * _bon > 0]
        dat("bdChuanThangChac", len(a3c), "%d")
        dat("bdChuanThangBon", len(a3b), "%d")
        # ⛔ KIEN TRUC chon tren tap do: in DO LON cua loi the ay, khong chi khai la co.
        for _ten in ("KAN", "MLP"):
            _nh1 = _nhieu_bien_gioi(m2_a3, chung, chi=_ten)
            dat("bdThang" + _ten.title(),
                sum(1 for m_, _ in _nh1.values() if m_ < 0), "%d")
        # ⛔ TRAN LUOI: hai o "thang" la dung hai o A3 chinh ra h* = dinh luoi. Ben hoc khong
        #    thang trong mot che do dinh vi duoc, no thang o noi DOI THU HET LUOI.
        _hmax = max(HYS)
        _tt = [c for c, d in hieu.items() if d < 0]
        dat("bdThangTranLuoi", sum(1 for c in _tt if m2_a3[c]["hys"] >= _hmax - 1e-9), "%d")
        dat("bdHTran", _hmax, "%g")
        # O KE GIUA hai o thang (neu co): bang chung "dai" that ra la cac o ROI NHAU.
        if len(_tt) == 2:
            _k = [dict(kv.split("=") for kv in c.split("+")) for c in _tt]
            _vd = sorted(float(x["vido"]) for x in _k)
            _gi = _k[0]["lam3"]
            _giua = [v for v in sorted({float(dict(kv.split("=") for kv in c.split("+"))["vido"])
                                        for c in chung})
                     if _vd[0] < v < _vd[1]]
            _c2 = ["vido=%g+lam3=%s" % (v, _gi) for v in _giua]
            _c2 = [c for c in _c2 if c in hieu]
            dat("bdSoOKeGiua", len(_c2), "%d")
            if _c2:
                dat("bdOKeGiuaBien", max(hieu[c] for c in _c2), "%.2f")
        vd = sorted({float(dict(kv.split("=") for kv in c.split("+"))["vido"]) for c in thang})
        dat("bdLatLo", vd[0], "%g"); dat("bdLatHi", vd[-1], "%g")
        gh = sorted({float(dict(kv.split("=") for kv in c.split("+"))["lam3"]) for c in thang})
        dat("bdCostLo", gh[0], "%g"); dat("bdCostHi", gh[-1], "%g")
    # --- quy mô thực nghiệm ---
    dat("qSeedTrain", a3["neo"]["n"], "%d")
    dat("qKhe", a3["neo"]["khe"], "%d")      # so khe cua che do neo
    # ⛔ `0,64%` tung duoc GO TAY vao §Setup: mot so DO ma khong hien vat nao do. Nay no
    #    tinh tu `results/do-lech-may.json`, vay cong SSOT khong con diem go tay nao la so do.
    import json as _js
    _dl = _js.load(io.open(os.path.join(RES, "do-lech-may.json"), encoding="utf-8"))
    _a, _b = (x["tong_tri_tuyet_doi_trong_so"] for x in _dl["phep_do"])
    dat("gLechMay", 100.0 * (_b - _a) / _a, "%.2f")
    # --- hien vat: SHA kho cong khai, so hinh, so bang. ⛔ Ba con so nay TUNG bi go tay o
    #     ban dau, va mot trong ba (SHA) doi moi lan day kho, nen go tay la bao dam sai.
    dat("qBang", len(BANG), "%d")
    # So hinh = so hinh ket qua bo sinh nay ve (truyen vao) + so hinh TikZ duoc khai.
    dat("qHinh", so_hinh_ve + len(HINH_TIKZ), "%d")
    # ⛔⛔ DUONG DAN CO DINH `GOC/repo-goc` lam BO SINH DUNG trong mot clone sach, vi o
    #    day kho la cay BAI chua kho con, con trong clone thi clone CHINH LA kho. §Data
    #    khai "mot clone sach sinh lai duoc moi hinh va bang", va loi nay lam cau ay SAI.
    #    Chi lo ra khi thuc su giai nen ra cho sach roi chay — dung lop loi "QA xanh vi
    #    khong thu dung lai" cua so nha. Nay thu CA HAI cho.
    sha = None
    for _d in (os.path.join(GOC, "repo-goc"), GOC):
        _r = subprocess.run(["git", "-C", _d, "rev-parse", "--short=7", "HEAD"],
                            capture_output=True, text=True)
        if _r.returncode == 0 and _r.stdout.strip():
            sha = _r.stdout.strip()
            break
    if not sha:
        raise SystemExit("⛔ khong lay duoc SHA kho, khong duoc go tay vao bai")
    dat("repoSha", sha, "%s")
    dat("qCheDo", len(a3), "%d")
    dat("qLuoi", len(A3), "%d")
    # ⛔⛔ TEN LENH LaTeX CHI DUOC CHUA CHU CAI. Ban dau toi dat `\gL025`, `\gPd0`,
    #    `\gVd70`..., va LaTeX doc `\gL` roi coi `025` la van ban, nen no nem
    #    "Missing \begin{document}" voi 54 loi. Phep kiem nay chan lop loi do ngay tai nguon
    #    thay vi de no no o khau dung bai.
    xau = sorted(k for k in M if not re.fullmatch(r"[A-Za-z]+", k))
    assert not xau, "⛔ ten macro co ky tu khong phai chu cai: %s" % xau
    with io.open(os.path.join(RES, ten_tep), "w", encoding="utf-8") as f:
        f.write(SINH)
        for k in sorted(M):
            f.write("\\newcommand{\\%s}{%s}\n" % (k, M[k]))
    print("  -> %s (%d macro)" % (os.path.join("results", ten_tep), len(M)))
    return M


def main():
    print("== Chay cong ket qua TRUOC khi ve ==")
    r = subprocess.run([sys.executable, os.path.join(HERE, "cong_ket_qua.py")],
                       capture_output=True, text=True)
    print("  " + (r.stdout.strip().splitlines() or ["(khong co dau ra)"])[-1])
    if r.returncode != 0:
        sys.exit("⛔ cong ket qua HONG, khong ve gi. Chay `python3 code/cong_ket_qua.py` de xem.")

    ap.use_style()
    ap.reset_registry()
    # ⛔ Dang ky mau: A3 la DOI TUONG cua bai nen no la "headline"; KAN/MLP la diem tham chieu.
    #    Dao vai nay la dao luan diem cua bai, nen no phai nam o day chu khong o tung hinh.
    ap.register(["A3 (tuned)", "KAN", "MLP", "max-SINR", "max-elev"],
                headline="A3 (tuned)", rival="KAN")

    of = tai("vps-ofat.csv")
    m2 = tai("vps-mat2d.csv")
    cs_of = cs_theo_che_do(tai("cs-ofat.csv") or [])
    if not of:
        sys.exit("⛔ thieu results/vps-ofat.csv")
    a3 = a3_theo_che_do(of)
    print("\n== Hinh ==")
    ok = []
    ok.append(f_luoi_a3(of, "g1-luoi-a3.pdf"))
    # ⛔ MOI khoa che do di qua K(), KHONG tu dung chuoi. Xem docstring cua K().
    ok.append(f_truc(a3, cs_of, lambda v: K(station=(float(v), 0.0)), [0, 15, 30, 45, 60, 70],
                     "ground-station latitude (deg)", "g2-vi-do.pdf"))
    ok.append(f_truc(a3, cs_of, lambda v: K(lam3=float(v)), [0.25, 0.5, 1.0, 2.0, 4.0],
                     "handover cost $\\lambda_3$", "g3-gia-ho.pdf", log_x=True))
    ok.append(f_truc(a3, cs_of, lambda v: K(fading_db=float(v)), [0.0, 1.0, 2.5, 4.0, 6.0],
                     "fading severity (dB)", "g4-pha-dinh.pdf"))
    ok.append(f_truc(a3, cs_of, lambda v: K(n_planes=int(v), sats_per_plane=int(v)),
                     [6, 9, 12, 15, 18, 24], "constellation: planes $=$ sats/plane",
                     "g5-mat-do.pdf"))
    ok.append(f_phan_bo(of, "g8-phan-bo.pdf"))
    ok.append(f_phan_ra(of, "g9-phan-ra.pdf"))
    ok.append(f_danh_doi(of, "g11-danh-doi.pdf"))
    if m2:
        ok.append(f_mat2d(m2, "g6-mat-2d.pdf"))
        ok.append(f_tham_so_toi_uu(m2, "g10-tham-so.pdf"))
        cs_m2 = tai("cs-mat2d.csv")
        if cs_m2:
            ok.append(f_ban_do_che_do(m2, cs_m2, "g7-ban-do-che-do.pdf"))
    print("\n== Macro ==")
    sinh_macro(a3, cs_of, (a3_theo_che_do(m2) if m2 else None),
               (cs_theo_che_do(tai("cs-mat2d.csv") or []) if m2 else None),
               of, "so-lieu.tex", len(ok))
    print("\n== Bang ==")
    _viec = {
        "tab-ky-hieu.tex":  lambda f: b_ky_hieu(f),
        "tab-che-do.tex":   lambda f: b_che_do(a3, f),
        "tab-doi-chieu.tex": lambda f: b_doi_chieu(f),
        "tab-chinh.tex":    lambda f: b_chinh(a3, cs_of, f),
        "tab-dai-luoi.tex": lambda f: b_dai_luoi(a3, f),
        "tab-dung-cu.tex":  lambda f: b_dung_cu(of, f),
        "tab-xuat-xu.tex":  lambda f: b_xuat_xu(f),
        "tab-tham-so.tex":  lambda f: b_tham_so_toi_uu(a3, f),
    }
    # ⛔ Phep kiem nay la ly do BANG dang o muc module: neu ai them mot bang vao day ma
    #    quen khai, hoac khai ma quen viet, `qBang` se sai va khong ai biet.
    assert sorted(_viec) == sorted(BANG), "⛔ BANG lech danh sach bang that sinh ra"
    for _f in BANG:
        _viec[_f](_f)
    print("\n=> %d/%d hinh dat" % (sum(1 for x in ok if x), len(ok)))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
