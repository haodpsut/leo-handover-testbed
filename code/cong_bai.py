#!/usr/bin/env python3
"""Cổng BÀI: kiểm bản thảo đã dựng, không kiểm ý định.

    python3 code/cong_bai.py            # kiểm bài
    python3 code/cong_bai.py --tu-kiem  # tiêm lỗi, mỗi ca phải làm cổng chuyển ĐẠT -> HỎNG

⛔ Mọi phép kiểm ở đây sinh ra từ một lần sai THẬT, phần lớn là sai của chính tôi ở bài này
hoặc ở bài IPM vừa nộp. Bốn luật đặt phép kiểm:

1. **Đọc MÃ TRẢ VỀ, không đọc nhãn.** Bản đầu của cổng hình kiểm `"PRINT-READY" in stdout`,
   mà dòng phán quyết là `PRINT-READY: FAIL`, nên nó báo ĐẠT trên đúng những hình đã bị bác.
2. **Kiểm trên NGUỒN khi hỏi về nguồn, trên PDF khi hỏi về trang in.** Ở bài IPM, phép kiểm
   ẩn danh đọc PDF cũ nên không thấy tên trường còn nằm trong tên tệp hình.
3. **Hai nguồn phải được SO với nhau.** Macro đúng không chứng minh câu văn quanh nó đúng,
   nên B-SO đối chứng từng số IN TRÊN TRANG với CSV, tính lại từ dữ liệu thô.
4. **Số đếm được thì máy đếm.** `\\nDeXuat` và mọi `\\g*`/`\\q*`/`\\bd*` phải do bộ sinh phát ra;
   một chữ số gõ tay trong thân bài là một lỗi, vì chạy lại thí nghiệm không cập nhật nó.
"""
import argparse
import glob
import io
import itertools
import os
import re
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOC = os.path.dirname(HERE)
PAPER = os.path.join(GOC, "paper")
RES = os.path.join(GOC, "results")
HYS, TTT = [0.0, 1.0, 2.0, 3.0, 4.0], [1, 2, 3]
A3 = ["a3_%g_%d" % (h, t) for h, t in itertools.product(HYS, TTT)]
HL_MAX = 85          # ky tu moi gach highlight, luat Elsevier
ABS_MAX = 250        # tu abstract


def doc(p):
    return io.open(p, encoding="utf-8").read()


def macro():
    v = {}
    for t in ("so-lieu.tex", "so-van-lieu.tex"):
        p = os.path.join(RES, t)
        if os.path.exists(p):
            v.update(dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", doc(p))))
    return v


def giai(van, mac):
    """Giải macro thành giá trị IN RA, rồi bỏ lệnh LaTeX, để đếm từ và ký tự như người đọc."""
    for k, val in sorted(mac.items(), key=lambda x: -len(x[0])):
        van = van.replace("\\" + k + "{}", val).replace("\\" + k, val)
    van = re.sub(r"\\begin\{\w+\}|\\end\{\w+\}", " ", van)
    van = re.sub(r"\\item", " ", van)
    van = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{?", " ", van)
    return re.sub(r"[{}$]|\\%", lambda m: "%" if m.group(0) == "\\%" else "", van)


def a3_tu_csv():
    """Tính lại khoảng cách A3 ĐÃ CHỈNH từ CSV THÔ, độc lập với bộ sinh macro."""
    import csv
    p = os.path.join(RES, "vps-ofat.csv")
    if os.path.exists(p):
        f_ = io.open(p, encoding="utf-8")
    elif os.path.exists(p + ".gz"):
        import gzip
        f_ = gzip.open(p + ".gz", "rt", encoding="utf-8")
    else:
        return {}
    g = {}
    for x in csv.DictReader(f_):
        g.setdefault((x["che_do"], x["vai"]), {}).setdefault(x["chinh_sach"], []).append(x)
    ra = {}
    for cd in {k[0] for k in g}:
        tr, te = g.get((cd, "train"), {}), g.get((cd, "test"), {})
        if not tr or not te:
            continue
        tb = {k: st.mean(float(y["gap_pct"]) for y in tr[k]) for k in A3 if k in tr}
        tot = min(tb, key=tb.get)
        ra[cd] = st.mean(float(y["gap_pct"]) for y in te[tot])
    return ra


def DANH_SACH_NGUON():
    """Mot nguon duy nhat cho danh sach tep nguon, dung boi ca `cham` va phep tiem."""
    return (sorted(glob.glob(os.path.join(PAPER, "sections", "*.tex")))
            + [os.path.join(PAPER, "main.tex"),
               os.path.join(PAPER, "frag-highlights.tex"),
               os.path.join(PAPER, "alg-dp.tex")])


def cham(tep=None):
    """tep: {duong_dan: noi_dung} de phep tiem thay noi dung ma khong ghi dia."""
    loi, luu = [], []

    def kiem(dk, ten, ct=""):
        print("  %s %s%s" % ("DAT " if dk else "HONG", ten, ("  [" + ct + "]") if ct else ""))
        if not dk:
            loi.append(ten)

    def chu_y(s):
        print("  chu y: %s" % s); luu.append(s)

    nguon = tep or {p: doc(p) for p in DANH_SACH_NGUON()}
    than = "\n".join(v for p, v in nguon.items() if "frag-" not in os.path.basename(p))

    # ---- B17 bo sinh hinh phai CHAY SACH ----
    # ⛔ Day la phep kiem bat duoc lop loi "hinh ve THIEU DIEM ma khong ai bao": `f_truc` nay
    #    tra False khi mot khoa che do khong khop, va bo sinh thoat ma 1. Mot hinh ve 2 tren 5
    #    diem da loi toi ban gui doc ngoai vi khong co cong nao chay lai bo sinh.
    _r = subprocess.run([sys.executable, os.path.join(HERE, "sinh_hinh_bang.py")],
                        capture_output=True, text=True)
    _d = [l for l in _r.stdout.splitlines() if "hinh dat" in l]
    kiem(_r.returncode == 0, "B17 bo sinh hinh chay sach",
         (_d[-1].strip() if _d else "ma tra ve %d" % _r.returncode))

    # ⛔⛔ BANG MACRO PHAI NAP SAU KHI BO SINH CHAY, khong truoc. Phep tiem chung minh:
    #    ca "tinh ti so o do chinh xac day du" sua BO SINH, nhung `mac` da nap tu tep cu nen
    #    B20 van doc 9,1 va bao DAT trong khi tep vua sinh ra 9,2. Dung lop loi "cong cham
    #    ban dung CU" da can o B16, va no quay lai ngay o phep kiem ke tiep vi toi chi doi
    #    THU TU HAI PHEP thay vi dat bo sinh len dau. Nay bo sinh chay TRUOC moi thu.
    mac = macro()

    # ---- B1 dung sach: dung lai tu nguon dang kiem, trong thu muc TAM ----
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="cong-bai-")
    try:
        shutil.copytree(PAPER, os.path.join(tmp, "paper"))
        shutil.copytree(RES, os.path.join(tmp, "results"),
                        ignore=shutil.ignore_patterns("*.csv", "*.json"))
        shutil.copytree(os.path.join(GOC, "figures"), os.path.join(tmp, "figures"))
        for p, v in nguon.items():
            q = os.path.join(tmp, os.path.relpath(p, GOC))
            io.open(q, "w", encoding="utf-8").write(v)
        d = os.path.join(tmp, "paper")
        for f in glob.glob(os.path.join(d, "main.*")):
            if not f.endswith(".tex"):
                os.remove(f)
        r = None
        for i in range(3):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                               cwd=d, capture_output=True, text=True)
            if i == 0:
                subprocess.run(["bibtex", "main"], cwd=d, capture_output=True, text=True)
        log = doc(os.path.join(d, "main.log")) if os.path.exists(os.path.join(d, "main.log")) else ""
        kiem(r.returncode == 0 and "! " not in log, "B1 dung lai SACH tu nguon",
             "ma tra ve %d, %d dong loi" % (r.returncode, log.count("\n! ")))
        kiem("Reference" not in log or "undefined" not in log,
             "B2 khong tham chieu treo",
             "%d ref + %d cite treo" % (len(re.findall(r"Reference.*undefined", log)),
                                        len(re.findall(r"Citation.*undefined", log))))
        kiem(not re.findall(r"Citation.*undefined", log), "B3 khong trich dan treo")
        kiem("Overfull" not in log, "B4 khong tran cot",
             "%d Overfull" % log.count("Overfull"))
        # ⛔ Khi mot ca tiem lam LaTeX hong NANG den muc khong ra PDF, cong phai ghi nhan
        #    HONG chu khong duoc nem ngoai le: mot cong chet giua chung thi khong phep kiem
        #    nao sau no chay, va bo tiem bao "that bai" thay vi "bat duoc". Khong ra PDF
        #    chinh la dang hong nang nhat, nen no phai di qua duong HONG binh thuong.
        import fitz
        _pp = os.path.join(d, "main.pdf")
        if os.path.exists(_pp):
            pdf = fitz.open(_pp)
            intr = "\n".join(pdf[i].get_text() for i in range(pdf.page_count))
            so_trang = pdf.page_count
            pdf.close()
        else:
            kiem(False, "B1b co sinh ra PDF", "khong co main.pdf sau 3 luot pdflatex")
            intr, so_trang = "", 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- B5 macro: moi macro bai dung phai TON TAI; va than bai khong go so tay ----
    dung = set(re.findall(r"\\((?:g|q|bd|n)[A-Z]\w*)", than))
    kiem(not (dung - set(mac)), "B5 moi macro bai dung deu do bo sinh phat ra",
         "thieu: %s" % sorted(dung - set(mac)) if dung - set(mac) else "%d macro" % len(dung))

    # ---- B6 KHONG go chu so vao cau ket qua ----
    # Chi soi 04-results/06-discussion: cac muc khac co tham so mo hinh, duoc phep co so.
    for ten in ("04-results.tex", "06-discussion.tex"):
        p = [q for q in nguon if q.endswith(ten)]
        if not p:
            continue
        v = re.sub(r"\\label\{[^}]*\}|\\ref\{[^}]*\}|\\eqref\{[^}]*\}|%[^\n]*", "", nguon[p[0]])
        v = re.sub(r"\$[^$]*\$", "", v)                      # cong thuc duoc phep co so
        v = re.sub(r"\\(?:cite|citep)[a-z]*\{[^}]*\}", "", v)
        # ⛔ BAO DONG GIA, sinh ra khi chuyen cac khoi float vao trong muc: `scale=1` cua
        #    `\includegraphics` va cac tuy chon `[tb]` cua moi truong KHONG phai so trong
        #    cau van, ma B6 van dem chung. Loai dung hai cau truc ay ra, chu KHONG ha phep
        #    kiem: ca tiem "go mot chu so vao cau ket qua" van phai bi bat.
        v = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", "", v)
        v = re.sub(r"\\(?:begin|end)\{\w+\*?\}(?:\[[^\]]*\])?", "", v)
        v = re.sub(r"\\input\{[^}]*\}", "", v)
        so = re.findall(r"(?<![\w\\^{])\d+(?:\.\d+)?(?![\w}])", v)
        so = [s for s in so if s not in ("10", "6", "2")]    # 10^{-6}, so muc, so mu
        kiem(not so, "B6 %s khong go chu so tay" % ten,
             "go tay: %s" % so[:6] if so else "0 so go tay")

    # ---- B7 doi chung BA TANG giua CSV, macro, va cau van ----
    # ⛔⛔ BAN DAU B7 chi hoi "so nay co xuat hien DAU DO trong PDF khong". Phep tiem chung
    #    minh no RONG NGHIA: doi `\gNeo` (0,76) thanh `\gNeoTeNhat` (24,3) thi cau van noi
    #    sai hoan toan, ma cong van DAT, vi 0,76 tinh co cung la gia tri cua `\gFadeMid` nen
    #    van con in o cho khac. Mot phep kiem "co mat o dau do" khong kiem duoc dieu no khai.
    #    Nay tach ba tang, va tang B7c moi la tang bat duoc ca tiem ay.
    goc = a3_tu_csv()
    NEO_CAU = {                 # (macro bat buoc, cum neo trong NGUON, tep)
        "neo": ("gNeo", r"\% from the unrestricted optimum", "04-results.tex"),
        "station=70.0": ("gLatPolar", r"at $\phi=70^\circ$, a factor of", "04-results.tex"),
        "station=15.0": ("gLatLow", r"The gap grows from", "04-results.tex"),
        "lam3=4.0": ("gCostHigh", r"the gap grows from", "04-results.tex"),
        "fading_db=0.0": ("gFadeZero", r"The gap is", "04-results.tex"),
    }
    if goc:
        # B7a macro phai khop CSV tinh lai bang CAI DAT DOC LAP
        # ⛔ KHONG co dinh "%.2f": mot so macro in o BA chu so (vi chung la can cu cho mot
        #    phep chia ma doc gia lam lai tren giay). Phep kiem phai so o DUNG do chinh xac
        #    macro khai, khong ap mot do chinh xac cua rieng no.
        def _nhu(m, v):
            n = len(mac[m].split(".")[-1]) if "." in mac[m] else 0
            return "%.*f" % (n, v)
        lech = [(m, mac.get(m), _nhu(m, goc[cd])) for cd, (m, _, _) in NEO_CAU.items()
                if cd in goc and m in mac and mac[m] != _nhu(m, goc[cd])]
        kiem(not lech, "B7a macro khop CSV tinh lai doc lap",
             "lech: %s" % lech if lech else "%d/%d khop" % (len(NEO_CAU), len(NEO_CAU)))
        # B7b gia tri ay phai that su IN RA (bat ca bai quen \input macro)
        thieu = [m for cd, (m, _, _) in NEO_CAU.items()
                 if cd in goc and mac.get(m) and mac[m] not in intr]
        kiem(not thieu, "B7b gia tri macro co in tren trang", "khong in: %s" % thieu)
        # B7c CAU VAN phai dung DUNG macro ay, khong phai mot macro khac cung dinh dang
        sai = []
        for cd, (m, cum, ten) in NEO_CAU.items():
            p = [q for q in nguon if q.endswith(ten)]
            if not p or cum not in nguon[p[0]]:
                sai.append((m, "khong thay cum neo")); continue
            # Cua so HAI PHIA: macro co the dung truoc cum neo ("sits \gNeo\% from the ...")
            # hoac sau ("The gap grows from \gLatLow\%"). Phep nay bat viec THAY macro trong
            # cau, khong bat duoc thu tu hai macro trong cung mot cau, va chi khai den day.
            i = nguon[p[0]].index(cum)
            quanh = nguon[p[0]][max(0, i - 110):i + len(cum) + 110]
            # ⛔ KHONG dung `("\\"+m) in quanh`: `\gNeoTeNhat` CHUA chuoi con `\gNeo`, nen
            #    phep khop chuoi cho qua dung ca tiem no phai bat. Phai co BIEN TU.
            if not re.search(r"\\%s(?![A-Za-z])" % re.escape(m), quanh):
                sai.append((m, re.findall(r"\\[a-z]{1,2}[A-Z]\w*", quanh)[-3:]))
        kiem(not sai, "B7c cau neo dung DUNG macro cua no",
             "sai: %s" % sai if sai else "%d/%d cau" % (len(NEO_CAU), len(NEO_CAU)))

    # ---- B8 em-dash, luat nha, ap cho MOI dau ra ----
    kiem("\u2014" not in than and "\u2014" not in intr, "B8 khong em-dash",
         "nguon %d, trang in %d" % (than.count("\u2014"), intr.count("\u2014")))

    # ---- B9 abstract: dem tu DA GIAI MACRO ----
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                  "\n".join(nguon.values()), re.S)
    if m:
        n = len(giai(m.group(1), mac).split())
        kiem(n <= ABS_MAX, "B9 abstract <= %d tu" % ABS_MAX, "%d tu" % n)
        con = re.findall(r"\\(?:g|q|bd|n)[A-Z]\w*", giai(m.group(1), mac))
        kiem(not con, "B9b abstract khong con macro chua giai", str(con[:3]))

    # ---- B10 highlights: <= 85 ky tu moi gach, DA GIAI MACRO ----
    hp = [q for q in nguon if "frag-highlights" in q]
    if hp:
        gach = [giai(x, mac).strip() for x in
                re.findall(r"\\item\s+(.+?)(?=\n\\item|\n\\end)", nguon[hp[0]], re.S)]
        gach = [re.sub(r"\s+", " ", g) for g in gach]
        dai = [(len(g), g[:40]) for g in gach if len(g) > HL_MAX]
        kiem(3 <= len(gach) <= 5, "B10 co 3 den 5 highlight", "%d gach" % len(gach))
        kiem(not dai, "B10b moi highlight <= %d ky tu" % HL_MAX,
             "qua dai: %s" % dai if dai else "dai nhat %d" % max(len(g) for g in gach))

    # ---- B11 cong thuc: moi phuong trinh co nhan phai duoc eqref it nhat mot lan ----
    nhan = set(re.findall(r"\\label\{(eq:[^}]*)\}", than))
    ref = set(re.findall(r"\\eqref\{(eq:[^}]*)\}", than))
    kiem(not (nhan - ref), "B11 khong phuong trinh mo coi",
         "mo coi: %s" % sorted(nhan - ref) if nhan - ref else "%d/%d duoc eqref"
         % (len(nhan), len(nhan)))

    # ---- B12 menh de phai co phep DO, khong chi co loi ----
    md = set(re.findall(r"\\begin\{proposition\}.*?\\label\{(prop:[^}]*)\}", than, re.S))
    rmd = set(re.findall(r"\\ref\{(prop:[^}]*)\}", than))
    kiem(not (md - rmd), "B12 moi menh de duoc dan lai o cho khac",
         "khong dan: %s" % sorted(md - rmd) if md - rmd else "%d menh de" % len(md))

    # ---- B13 hinh va bang: moi tep hinh duoc dung, moi nhan duoc ref ----
    hinh = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{\.\./figures/([^}]*)\}", than))
    tren_dia = {os.path.basename(p) for p in glob.glob(os.path.join(GOC, "figures", "*.pdf"))}
    kiem(hinh <= tren_dia, "B13 moi hinh bai dung deu co tren dia",
         "thieu: %s" % sorted(hinh - tren_dia) if hinh - tren_dia else "%d hinh" % len(hinh))
    if tren_dia - hinh:
        chu_y("hinh tren dia ma bai KHONG dung: %s" % sorted(tren_dia - hinh))
    for k in ("fig", "tab"):
        n_ = set(re.findall(r"\\label\{(%s:[^}]*)\}" % k, than))
        r_ = set(re.findall(r"\\ref\{(%s:[^}]*)\}" % k, than))
        kiem(not (n_ - r_), "B13%s moi %s duoc dan trong van" % ("c" if k == "fig" else "d", k),
             "khong dan: %s" % sorted(n_ - r_) if n_ - r_ else "%d/%d" % (len(n_), len(n_)))

    # ---- B14 moi hop thuat toan phai duoc dan trong van ----
    nalg = set(re.findall(r"\\label\{(alg:[^}]*)\}", than))
    ralg = set(re.findall(r"\\ref\{(alg:[^}]*)\}", than))
    kiem(not (nalg - ralg), "B14 moi thuat toan duoc dan trong van",
         "khong dan: %s" % sorted(nalg - ralg) if nalg - ralg else "%d/%d" % (len(nalg), len(nalg)))
    kiem(len(nalg) >= 1, "B14b bai co hop thuat toan", "%d hop" % len(nalg))

    # ⛔⛔ THU TU NAY QUAN TRONG, va phep tiem day no ra. B17 (chay lai bo sinh) PHAI dung
    #    TRUOC B16 (soi hinh tren dia). Ban dau nguoc lai, nen khi tiem loi `\%` vao bo sinh
    #    thi B16 doc cac PDF CU con sach va bao DAT, roi B17 moi ve lai hinh hong. Cong logic
    #    dung ma cham ban dung CU: lop loi 32 cua so nha, lap lai y nguyen o day.
    # ---- B16 KHONG `\%` literal trong HINH ----
    # ⛔ usetex TAT, nen `"gap (\\%)"` trong matplotlib in ra DAU GACH CHEO tren giay. Ba hinh
    #    da di toi tan ban gui doc ngoai voi loi nay, va `measure_plot` khong bat vi no do co
    #    chu chu khong doc chu. Hao bat bang mat.
    import fitz as _f
    xau_pc = []
    for q in sorted(glob.glob(os.path.join(GOC, "figures", "*.pdf"))):
        _d = _f.open(q)
        if "\\%" in "\n".join(_d[i].get_text() for i in range(_d.page_count)):
            xau_pc.append(os.path.basename(q))
        _d.close()
    kiem(not xau_pc, "B16 khong dau gach cheo truoc % trong hinh",
         "co: %s" % xau_pc if xau_pc else "%d hinh sach"
         % len(glob.glob(os.path.join(GOC, "figures", "*.pdf"))))

    # ---- B18 HINH LUONG phai qua cong hinh luong ----
    # ⛔ Hinh 1 tung la TikZ noi tuyen, nen KHONG cong nao cham no: `measure_plot` do hinh
    #    ket qua, con `measure_figure` (cong cua flow figure) thi khong duoc goi bao gio. No
    #    di toi tan ban gui doc ngoai voi card DE LEN NHAU. Nay no la mot PDF dung rieng va
    #    phai qua dung cong cua no, giong moi hinh khac.
    _mf = os.path.join(GOC, "..", "..", "my-skills", "ALL-FIGURES", "0-SKILLS",
                       "draw-figure-dph", "assets", "measure_figure.py")
    _pdf = os.path.join(GOC, "figures", "g0-luong.pdf")
    if os.path.exists(_mf) and os.path.exists(_pdf):
        import fitz as _f2
        import tempfile as _tf
        _dd = _tf.mkdtemp(prefix="cong-luong-")
        try:
            _d2 = _f2.open(_pdf)
            _d2[0].get_pixmap(dpi=330).save(os.path.join(_dd, "g0-luong.png"))
            _d2.close()
            shutil.copy(_pdf, os.path.join(_dd, "g0-luong.pdf"))
            _r2 = subprocess.run([sys.executable, _mf, os.path.join(_dd, "g0-luong.png"),
                                  "--column", "single"], capture_output=True, text=True)
            _l = [x for x in _r2.stdout.splitlines() if x.startswith("COMPOSITION")]
            kiem(_r2.returncode == 0, "B18 hinh luong qua cong measure_figure",
                 (_l[-1] if _l else "ma tra ve %d" % _r2.returncode))
        finally:
            shutil.rmtree(_dd, ignore_errors=True)
    else:
        chu_y("B18 bo qua: khong thay measure_figure.py hoac g0-luong.pdf")

    # ---- B19 MOI SO TRONG BANG phai tinh lai duoc tu CSV, bang mot cai dat KHAC ----
    # ⛔⛔ Phan bien ngoai 11/10 bat mot bang TOM TAT in sai vi bo sinh dung khoa che do sai
    #    chinh ta, va hau qua la bang DAO NGUOC thu hang cac truc so voi ket luan cua bai.
    #    Cong cu chi hoi "macro co ton tai khong" va "so co in ra khong", khong hoi "con so
    #    TONG HOP nay co dung khong". Nay tinh lai tung o cua bang truc che do tu CSV tho,
    #    bang mot cai dat doc lap, va so tung o.
    import csv as _csv
    _p = os.path.join(RES, "vps-ofat.csv")
    _bang = os.path.join(RES, "tables", "tab-che-do.tex")
    if os.path.exists(_p) and os.path.exists(_bang):
        _g = {}
        for x in _csv.DictReader(io.open(_p, encoding="utf-8")):
            _g.setdefault((x["che_do"], x["vai"]), {}).setdefault(x["chinh_sach"], []).append(x)

        def _gap(cd):
            tr, te = _g[(cd, "train")], _g[(cd, "test")]
            tb = {k: st.mean(float(y["gap_pct"]) for y in tr[k]) for k in A3 if k in tr}
            return st.mean(float(y["gap_pct"]) for y in te[min(tb, key=tb.get)])

        _truc = {
            "latitude": ["station=%.1f" % v for v in (0., 15., 30., 45., 60., 70.)],
            "handover cost": ["neo" if v == 1.0 else "lam3=%s" % v
                              for v in (0.25, 0.5, 1.0, 2.0, 4.0)],
            "fading": ["neo" if v == 2.5 else "fading_db=%.1f" % v
                       for v in (0., 1., 2.5, 4., 6.)],
            "slot": ["neo" if d == 15 else "S=%d+slot_seconds=%d" % (int(1800 / d), d)
                     for d in (5, 15, 30, 60)],
            "planes": ["neo" if k == 18 else "n_planes=%d+sats_per_plane=%d" % (k, k)
                       for k in (6, 9, 12, 15, 18, 24)],
        }
        _van = doc(_bang)
        _xau = []
        for _ten, _cds in _truc.items():
            _v = [_gap(c) for c in _cds if (c, "test") in _g]
            if len(_v) != len(_cds):
                _xau.append((_ten, "thieu %d khoa" % (len(_cds) - len(_v)))); continue
            _dong = [l for l in _van.splitlines() if l.startswith(_ten.split()[0])]
            if not _dong:
                _xau.append((_ten, "khong thay hang")); continue
            _so = re.findall(r"(\d+\.\d+)", _dong[0].split("&", 2)[2])
            _mong = ["%.2f" % min(_v), "%.2f" % max(_v), "%.2f" % (max(_v) - min(_v))]
            if _so[-3:] != _mong:
                _xau.append((_ten, "bang in %s, tinh lai ra %s" % (_so[-3:], _mong)))
        kiem(not _xau, "B19 moi o cua bang truc che do khop CSV tinh lai",
             ("lech: %s" % _xau) if _xau else "%d truc khop" % len(_truc))

    # ---- B20 TI SO IN RA phai bang ti so CUA CAC SO IN RA ----
    # ⛔ Bai in 2,91 va 0,32 roi in ti so 9,2; nguoi doc chia tay ra 9,1. Mot so khong kiem
    #    lai duoc tu chinh trang giay la mot so khong kiem duoc.
    _ts = [("gLatPolarRatio", "gLatPolar", "gLatLow"),
           ("gCostRatio", "gCostHigh", "gCostLow"),
           ("gFadeZeroTi", "gFadeZeroFloor", "gFadeZero")]
    _xau2 = []
    for _r, _tu, _mau in _ts:
        if not all(k in mac for k in (_r, _tu, _mau)):
            continue
        # ⛔⛔ BAN DAU PHEP NAY HOI "ti so co nam trong DAI lam tron khong". Phep tiem chung
        #    minh no YEU: dai cua 2,91/0,32 la [8,94; 9,25], nen con so SAI 9,2 van lot, dung
        #    cai so phan bien ngoai bat. Mot dieu kien CAN khong thay duoc dieu kien DU. Nay
        #    hoi dung cai doc gia lam: chia hai so in ra, lam tron ve do chinh xac cua ti so.
        _k = 100.0 if _r.endswith("Ti") else 1.0
        _n = len(mac[_r].split(".")[-1]) if "." in mac[_r] else 0
        _mong = "%.*f" % (_n, round(_k * float(mac[_tu]) / float(mac[_mau]), _n))
        if mac[_r] != _mong:
            _xau2.append((_r, "bai in %s, chia tay ra %s" % (mac[_r], _mong)))
    kiem(not _xau2, "B20 ti so in ra BANG ti so cua cac so in ra",
         ("lech: %s" % _xau2) if _xau2 else "%d ti so khop" % len(_ts))

    # ---- B15 khong hien vat nao chi ton tai mot noi ----
    # Mot bang sinh ra ma main.tex khong \input la mot bang chet; cong phai thay.
    import glob as _g
    tab_dia = {os.path.basename(q)[:-4] for q in _g.glob(os.path.join(RES, "tables", "*.tex"))}
    tab_dung = set(re.findall(r"\\input\{\.\./results/tables/([^}]*)\}", than))
    kiem(not (tab_dia - tab_dung), "B15 moi bang sinh ra deu duoc \\input",
         "bang chet: %s" % sorted(tab_dia - tab_dung) if tab_dia - tab_dung
         else "%d bang" % len(tab_dia))

    print("\n=> %s (%d loi, %d chu y)  TRANG: %d"
          % ("DAT" if not loi else "CHUA DAT", len(loi), len(luu), so_trang))
    return loi


def tu_kiem():
    # ⛔ Danh sach nay PHAI trung danh sach trong `cham()`. Lan dau no thieu fig-luong.tex va
    #    alg-dp.tex nen ban GOC cua phep tiem tu HONG (B13c, B14), tuc khung tiem bao dong gia
    #    tren chinh no. Nay lay chung mot nguon de khong the lech nua.
    goc = {p: doc(p) for p in DANH_SACH_NGUON()}
    kq = os.path.join(PAPER, "sections", "04-results.tex")
    hl = os.path.join(PAPER, "frag-highlights.tex")
    mt = os.path.join(PAPER, "main.tex")
    f3 = os.path.join(PAPER, "sections", "03-formulation.tex")
    st_ = os.path.join(PAPER, "sections", "03b-setup.tex")

    def doi(p, old, new):
        d = dict(goc)
        assert old in d[p], "⛔ ca tiem khong khop: %s trong %s" % (old[:40], os.path.basename(p))
        d[p] = d[p].replace(old, new, 1)
        assert d[p] != goc[p]
        return d

    ca = [
        # ⛔ Ca nay PHAI khong co chu so, neu khong B6 bat truoc va B7 van chua duoc chung minh.
        #    Doi \gNeo thanh mot macro KHAC: so in ra khong con khop CSV, ma khong go so tay nao.
        ("B7: doi so in ra sang macro khac, khong go chu so",
         lambda: doi(kq, r"\gNeo\% from the unrestricted optimum",
                     r"\gNeoTeNhat\% from the unrestricted optimum")),
        ("B8: tiem mot em-dash",
         lambda: doi(kq, "The instrument:", "The instrument\u2014")),
        ("B5: dung macro KHONG ton tai",
         lambda: doi(kq, r"\gNeoVis{} satellites", r"\gKhongTonTai{} satellites")),
        ("B6: go mot chu so vao cau ket qua",
         lambda: doi(kq, "The more useful number", "The 42 more useful number")),
        ("B10b: lam mot highlight dai hon 85 ky tu",
         lambda: doi(hl, r"\item Learned policies beat tuned A3 in only \bdWins{} of \bdCells{} regime cells",
                     r"\item Learned policies beat the tuned standard A3 mechanism in only "
                     r"\bdWins{} of \bdCells{} regime cells of the latitude by handover cost slice")),
        ("B9: lam abstract dai hon %d tu" % ABS_MAX,
         lambda: doi(mt, r"\end{abstract}", ("x " * 120) + "\n\\end{abstract}")),
        # ⛔ Phai bo eqref cua phuong trinh chi duoc dan DUNG MOT LAN, va thay bang mot thu
        #    KHONG co chu so, de chinh B11 bat chu khong phai B6.
        # ⛔ Neo cu la eq:decomp, nhung sau khi them hinh luong thi no duoc eqref o HAI cho,
        #    nen bo mot cho khong con mo coi va ca tiem im lang. Phai neo vao phuong trinh
        #    duoc dan DUNG MOT LAN. (Day la lan thu hai neo tiem muc ruong theo bai.)
        ("B11: bo eqref duy nhat cua eq:branch",
         lambda: doi(st_, r"the bound \eqref{eq:branch} turns",
                     r"the branching bound turns")),
        # ⛔ HAI CA NAY tai hien DUNG hai loi Hao bat bang mat ngay 11/10, va chung tiem vao
        #    BO SINH chu khong vao ban thao, vi do la noi loi that da nam.
        ("B17: lam mot khoa che do sai chinh ta trong bo sinh",
         None),
        ("B16: tra lai `\\%` literal trong nhan hinh",
         None),
        ("B19: tra lai khoa `%g` cho bang truc che do", None),
        ("B20: tinh ti so o do chinh xac day du thay vi tu so in ra", None),
        ("B2: tiem mot \\ref treo",
         lambda: doi(kq, r"Table~\ref{tab:dai}", r"Table~\ref{tab:khong-co}")),
    ]
    print("== Ban GOC ==")
    if cham(goc):
        sys.exit("⛔ ban GOC da hong, khong tiem duoc")
    hong = []
    bs = os.path.join(HERE, "sinh_hinh_bang.py")
    ma_goc = doc(bs)
    tiem_ma = {
        # ⛔ BA CA DUOI DAY tai hien DUNG ba loi phan bien ngoai tim ra ngay 11/10, va chung
        #    tiem vao BO SINH vi do la noi ba loi that da nam.
        "B19: tra lai khoa `%g` cho bang truc che do":
            ('[K(lam3=v) for v in (0.25, 0.5, 1., 2., 4.)]',
             '[("lam3=%g" % v) for v in (0.25, 0.5, 1., 2., 4.)]'),
        "B20: tinh ti so o do chinh xac day du thay vi tu so in ra":
            ('dat("gLatPolarRatio", _ti(a3[K(station=(70.0, 0.0))]["gap"],\n'
             '                              a3[K(station=(15.0, 0.0))]["gap"]), "%.1f")',
             'dat("gLatPolarRatio", a3[K(station=(70.0, 0.0))]["gap"]\n'
             '        / a3[K(station=(15.0, 0.0))]["gap"], "%.1f")'),
        "B17: lam mot khoa che do sai chinh ta trong bo sinh":
            ('lambda v: K(lam3=float(v))', 'lambda v: "lam3=%g" % v'),
        "B16: tra lai `\\%` literal trong nhan hinh":
            ('ax.set_ylabel("optimality gap (%)")', 'ax.set_ylabel("optimality gap (\\\\%)")'),
    }
    for ten, f in ca:
        print("\n== %s ==" % ten)
        if ten in tiem_ma:
            cu, moi = tiem_ma[ten]
            assert cu in ma_goc, "⛔ ca tiem khong khop ma: %s" % cu[:40]
            io.open(bs, "w", encoding="utf-8").write(ma_goc.replace(cu, moi, 1))
            try:
                l = cham()
            finally:
                io.open(bs, "w", encoding="utf-8").write(ma_goc)
                subprocess.run([sys.executable, bs], capture_output=True, text=True)
            print("  => %s" % ("cong BAT duoc" if l else "⛔ cong KHONG bat"))
            if not l:
                hong.append(ten)
            continue
        l = cham(f())
        print("  => %s" % ("cong BAT duoc" if l else "⛔ cong KHONG bat"))
        if not l:
            hong.append(ten)
    print("\n=> %s (%d/%d ca)" % ("DAT" if not hong else "CHUA DAT",
                                  len(ca) - len(hong), len(ca)))
    for h in hong:
        print("   HONG: %s" % h)
    return 1 if hong else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()
    if a.tu_kiem:
        return tu_kiem()
    print("== Cong bai ==\n")
    return 1 if cham() else 0


if __name__ == "__main__":
    sys.exit(main())
