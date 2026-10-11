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
                      vis=m(tot, "vis_per_slot"), n=len(te[tot]))
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
    co = [c for c in cd if c in a3]
    if len(co) < 2:
        print("  bo qua %s: chi %d diem" % (ten_tep, len(co)))
        return False
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
    if log_x:
        # ⛔ `set_xscale("log")` sinh nhan dang `10^{0}`, va so mu mathtext ve o ~70% co chu
        #    nen ra 4,9pt, duoi san 7,0pt. Dat nhan TUONG MINH thi khong co so mu nao.
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([("%g" % v) for v in x])
        ax.minorticks_off()
    ax.set_xlabel(nhan_x)
    ax.set_ylabel("optimality gap (\\%)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    return luu(fig, ten_tep)


def f_luoi_a3(hang, ten_tep):
    """Bản đồ nhiệt khoảng cách trên lưới (hys, TTT), ở chế độ neo."""
    te = [x for x in hang if x["che_do"] == "neo" and x["vai"] == "test"]
    if not te:
        print("  bo qua %s: khong co che do neo" % ten_tep)
        return False
    m = np.full((len(TTT), len(HYS)), np.nan)
    for i, t in enumerate(TTT):
        for j, h in enumerate(HYS):
            v = [float(x["gap_pct"]) for x in te if x["chinh_sach"] == "a3_%g_%d" % (h, t)]
            if v:
                m[i, j] = st.mean(v)
    fig, ax = plt.subplots(figsize=(W1, 2.0), constrained_layout=True)
    im = ax.imshow(m, aspect="auto", origin="lower", cmap="YlOrRd")
    ax.set_xticks(range(len(HYS))); ax.set_xticklabels(["%g" % h for h in HYS])
    ax.set_yticks(range(len(TTT))); ax.set_yticklabels([str(t) for t in TTT])
    ax.set_xlabel("hysteresis (dB)"); ax.set_ylabel("time-to-trigger (slots)")
    for i in range(len(TTT)):
        for j in range(len(HYS)):
            if not np.isnan(m[i, j]):
                # ⛔ fontsize=6 ra 6,1pt tren giay, duoi san 7,0pt cua `measure_plot`.
                #    San ay khong phai so tron tay: duoi no thi chu khong doc duoc khi in.
                ax.text(j, i, "%.1f" % m[i, j], ha="center", va="center", fontsize=7,
                        color=("white" if m[i, j] > np.nanmax(m) * 0.6 else ap.INK))
    fig.colorbar(im, ax=ax, label="gap (\\%)")
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
    im = ax.pcolormesh(vi, gh, m, shading="nearest", cmap="YlOrRd")
    ax.set_yscale("log")
    ax.set_xlabel("ground-station latitude (deg)")
    ax.set_ylabel("handover cost $\\lambda_3$")
    fig.colorbar(im, ax=ax, label="A3 gap (\\%)")
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
    im = ax.pcolormesh(vi, gh, m, shading="nearest", cmap="RdBu_r", norm=norm)
    ax.set_yscale("log")
    ax.set_yticks(gh); ax.set_yticklabels([("%g" % v) for v in gh]); ax.minorticks_off()
    ax.set_xlabel("ground-station latitude (deg)")
    ax.set_ylabel("handover cost $\\lambda_3$")
    fig.colorbar(im, ax=ax, label="learned $-$ tuned A3 (\\%)")
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
    ax.set_ylabel("tuned A3 gap (\\%)")
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
    ax.set_ylabel("tuned A3 gap (\\%)")
    ax.legend(frameon=False, fontsize=7.0, loc="upper left")
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
    if len(vi) < 2 or len(gh) < 2:
        print("  bo qua %s" % ten_tep); return False
    m = np.full((len(gh), len(vi)), np.nan)
    for i, c in enumerate(gh):
        for j, v in enumerate(vi):
            cd = "vido=%g+lam3=%g" % (v, c)
            if cd in a3h:
                m[i, j] = a3h[cd]["hys"]
    fig, ax = plt.subplots(figsize=(W1, 2.3), constrained_layout=True)
    im = ax.pcolormesh(vi, gh, m, shading="nearest", cmap="viridis",
                       vmin=min(HYS), vmax=max(HYS))
    ax.set_yscale("log")
    ax.set_yticks(gh); ax.set_yticklabels([("%g" % v) for v in gh]); ax.minorticks_off()
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
    ax.set_ylabel("tuned A3 gap (\\%)")
    ax.set_yscale("log")
    ax.set_yticks([1, 3, 10, 30]); ax.set_yticklabels(["1", "3", "10", "30"])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=7.0, loc="upper right")
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
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Regime & \\multicolumn{2}{c}{tuned A3} & floor & KAN & MLP & max-SINR \\\\\n")
        f.write("\\cmidrule(lr){2-3}\n")
        f.write(" & $(h,T)$ & gap & & gap & gap & gap \\\\\n\\midrule\n")
        for cd, ten in thu_tu:
            if cd not in a3:
                continue
            v = a3[cd]
            k = cs.get(cd, {})
            f.write("%s & (%g, %d) & %.2f & %.2f & %s & %s & %.2f \\\\\n"
                    % (ten, v["hys"], v["ttt"], v["gap"], v["gia"],
                       ("%.2f" % k["KAN"]) if "KAN" in k else "n/a",
                       ("%.2f" % k["MLP"]) if "MLP" in k else "n/a", v["msinr"]))
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
    truc = [
        ("latitude $\\phi$ (deg)", "0, 15, 30, 45, 60, 70",
         [("station=%.1f" % v) for v in (0., 15., 30., 45., 60., 70.)]),
        ("handover cost $\\lam_3$", "0.25, 0.5, 1, 2, 4",
         [("lam3=%g" % v) for v in (0.25, 0.5, 1., 2., 4.)]),
        ("fading $\\sigma$ (dB)", "0, 1, 2.5, 4, 6",
         [("fading_db=%.1f" % v) for v in (0., 1., 2.5, 4., 6.)]),
        ("slot $\\Delta$ (s)", "5, 15, 30, 60",
         [("S=%d+slot_seconds=%d" % (int(1800 / d), d)) for d in (5, 15, 30, 60)]),
        ("planes $=$ sats/plane", "6, 9, 12, 15, 18, 24",
         [("n_planes=%d+sats_per_plane=%d" % (k, k)) for k in (6, 9, 12, 15, 18, 24)]),
    ]
    with io.open(os.path.join(TAB, ten_tep), "w", encoding="utf-8") as f:
        f.write(SINH)
        f.write("\\begin{tabular}{llrrr}\n\\toprule\n")
        f.write("Axis & Values swept & min gap & max gap & span \\\\\n\\midrule\n")
        for ten, gt, cds in truc:
            v = [a3[c]["gap"] for c in cds if c in a3]
            if not v:
                continue
            f.write("%s & %s & %.2f & %.2f & \\textbf{%.2f} \\\\\n"
                    % (ten, gt, min(v), max(v), max(v) - min(v)))
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
        ("\\citet{zhao2026twc}", [Q, N, N, N]),
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


def sinh_macro(a3, cs, m2_a3, m2_cs, of_hang, ten_tep):
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
    # --- vĩ độ ---
    dat("gLatLow", a3[K(station=(15.0, 0.0))]["gap"])
    dat("gLatPolar", a3[K(station=(70.0, 0.0))]["gap"])
    dat("gLatPolarFloor", a3[K(station=(70.0, 0.0))]["gia"])
    dat("gLatPolarRatio", a3[K(station=(70.0, 0.0))]["gap"] / a3[K(station=(15.0, 0.0))]["gap"], "%.1f")
    # ⭐ CƠ CHẾ phía sau trục mạnh nhất. ⛔ GIẢI THÍCH ĐẦU CỦA TÔI SAI và số liệu bác ngay:
    #    tôi viết "vì 70° THIẾU ỨNG VIÊN (1,82/khe)". Nhưng chế độ mật độ 6 còn thiếu hơn
    #    (1,20/khe) mà khoảng cách chỉ 0,41%, tức thiếu ứng viên MỘT MÌNH không giải thích gì.
    #    Thứ phân biệt hai chế độ là LUÂN CHUYỂN: ở 70° có 38,6 lần chuyển giao và 11,45
    #    ping-pong mỗi 120 khe, còn ở mật độ 6 chỉ 9,2 và 1,40. Thiếu ứng viên CỘNG luân chuyển
    #    nhanh mới là cơ chế, và mật độ 6 là ĐỐI CHỨNG cho đúng câu ấy. Giữ cả hai bộ số để
    #    §Results in được cả đối chứng, chứ không chỉ in cái khớp với câu mình muốn viết.
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
    dat("gCostRatio", a3[K(lam3=4.0)]["gap"] / a3[K(lam3=0.25)]["gap"], "%.1f")
    dat("gCostHighGreedy", a3[K(lam3=4.0)]["msinr"])
    dat("gHysDich", "%g, %g, %g, %g, %g" % tuple(
        a3[K(lam3=c)]["hys"] for c in (0.25, 0.5, 1.0, 2.0, 4.0)))
    # --- pha đinh ---
    dat("gFadeZero", a3[K(fading_db=0.0)]["gap"]); dat("gFadeZeroFloor", a3[K(fading_db=0.0)]["gia"])
    dat("gFadeZeroLearn", cs[K(fading_db=0.0)]["KAN"], "%.3f")
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
    ti = sorted(100.0 * a3[c]["gia"] / a3[c]["gap"] for c in a3 if a3[c]["gap"] > 1e-9)
    dat("gSanTiTv", ti[len(ti) // 2], "%.1f")
    dat("gSanTiLo", max(0.0, ti[0]), "%.1f")      # ⛔ kep ve 0: tỉ lệ âm là vô nghĩa, và nó
    dat("gSanTiHi", ti[-1], "%.1f")               #    chỉ sinh ra từ làm tròn -0.000000
    dat("gSanCheDo", len(ti), "%d")
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
    _cd = [c for c in a3 if a3[c]["gap"] > 1e-9]
    dat("gSanTuongQuanVis", _pear([a3[c]["gia"] for c in _cd], [a3[c]["vis"] for c in _cd]), "%.2f")
    dat("gSanLonNhatMong", a3[K(station=(70.0, 0.0))]["gia"], "%.3f")
    dat("gSanNhoNhatDay", a3[K(station=(15.0, 0.0))]["gia"], "%.3f")
    dat("gVisMong", a3[K(station=(70.0, 0.0))]["vis"], "%.1f")
    dat("gVisDay", a3[K(station=(15.0, 0.0))]["vis"], "%.1f")
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
    dat("qDongTong", sum(len(tai(t) or []) for t in
                         ("vps-ofat.csv", "vps-mat2d.csv", "cs-ofat.csv", "cs-mat2d.csv")), "%d")
    # --- bản đồ chế độ ---
    if m2_a3 and m2_cs:
        chung = [c for c in m2_a3 if c in m2_cs and m2_cs[c]]
        hieu = {c: min(m2_cs[c].values()) - m2_a3[c]["gap"] for c in chung}
        thang = [c for c, d in hieu.items() if d < 0]
        dat("bdCells", len(chung), "%d"); dat("bdWins", len(thang), "%d")
        dat("bdWinPct", 100.0 * len(thang) / len(chung), "%.1f")
        dat("bdBestMargin", -min(hieu.values()), "%.2f")
        dat("bdWorstMargin", max(hieu.values()), "%.2f")
        vd = sorted({float(dict(kv.split("=") for kv in c.split("+"))["vido"]) for c in thang})
        dat("bdLatLo", vd[0], "%g"); dat("bdLatHi", vd[-1], "%g")
        gh = sorted({float(dict(kv.split("=") for kv in c.split("+"))["lam3"]) for c in thang})
        dat("bdCostLo", gh[0], "%g"); dat("bdCostHi", gh[-1], "%g")
    # --- quy mô thực nghiệm ---
    dat("qSeedTrain", a3["neo"]["n"], "%d")
    dat("qKhe", a3["neo"]["khe"], "%d")      # so khe cua che do neo      # n của test; train cùng cỡ
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
    ok.append(f_truc(a3, cs_of, lambda v: "station=%.1f" % v, [0, 15, 30, 45, 60, 70],
                     "ground-station latitude (deg)", "g2-vi-do.pdf"))
    ok.append(f_truc(a3, cs_of, lambda v: "lam3=%g" % v, [0.25, 0.5, 1.0, 2.0, 4.0],
                     "handover cost $\\lambda_3$", "g3-gia-ho.pdf", log_x=True))
    ok.append(f_truc(a3, cs_of, lambda v: "fading_db=%.1f" % v, [0.0, 1.0, 2.5, 4.0, 6.0],
                     "fading severity (dB)", "g4-pha-dinh.pdf"))
    ok.append(f_truc(a3, cs_of, lambda v: "n_planes=%d+sats_per_plane=%d" % (v, v),
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
               of, "so-lieu.tex")
    print("\n== Bang ==")
    b_ky_hieu("tab-ky-hieu.tex")
    b_che_do(a3, "tab-che-do.tex")
    b_doi_chieu("tab-doi-chieu.tex")
    b_chinh(a3, cs_of, "tab-chinh.tex")
    b_dai_luoi(a3, "tab-dai-luoi.tex")
    print("\n=> %d/%d hinh dat" % (sum(1 for x in ok if x), len(ok)))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
