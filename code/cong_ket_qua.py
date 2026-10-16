#!/usr/bin/env python3
"""Cổng kiểm KẾT QUẢ, chạy trước khi tin một con số nào.  python3 code/cong_ket_qua.py

⛔ Một hình vẽ từ số chưa qua cổng là một hình phải vẽ lại. Nên cổng này đứng TRƯỚC khâu vẽ.

Sáu phép, và mỗi phép kiểm một thứ có thể hỏng ÂM THẦM:

  K1  đối chứng dương: `a3_0_1` phải TRÙNG KHÍT `max-sinr` ở mọi dòng. Lệch là bộ đo hỏng,
      không phải một phát hiện. Đây là phép duy nhất có ngưỡng tuyệt đối 0.
  K2  thứ tự DP: không chính sách nào được tốt hơn DP đầy đủ ⇒ `gap_pct >= 0` ở mọi dòng.
  K3  giá phép rút gọn nhị phân: DP nhị phân là một RÀNG BUỘC của DP đầy đủ nên không được
      tốt hơn.
  K4  mệnh đề suy biến: ở `fading_db=0`, A3 tốt nhất phải CHÍNH LÀ `hys=0, ttt=1`. Bài phát
      biểu điều này thành một mệnh đề, nên nó phải được ĐO chứ không được khẳng định.
  K5  đủ đơn vị: số dòng phải bằng số đơn vị × số chính sách. Thiếu âm thầm là mất dữ liệu.
  K6  MỘT máy: mọi tệp `.xuat-xu.json` phải cùng host, cùng phiên bản gói, cùng băm mã.
      Luật này ở bài này không hình thức: đo được MLP lệch 0,64% giữa hai host.
"""
import collections
import csv
import glob
import io
import itertools
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOC = os.path.dirname(HERE)
RES = os.path.join(GOC, "results")
loi, chuy = [], []


def kiem(dk, ten, ct=""):
    print("  %s %s%s" % ("DAT " if dk else "HONG", ten, ("  [" + ct + "]") if ct else ""))
    if not dk:
        loi.append(ten)
    return dk


def luu_y(ten, ct=""):
    print("  CHUY %s%s" % (ten, ("  [" + ct + "]") if ct else ""))
    chuy.append(ten)


def doc(ten):
    """Doc CSV, nhan ca ban nen `.csv.gz` nhu trong kho cong khai."""
    p = os.path.join(RES, ten)
    if os.path.exists(p):
        with io.open(p, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if os.path.exists(p + ".gz"):
        import gzip
        with gzip.open(p + ".gz", "rt", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return None


def main():
    print("== Cong kiem ket qua ==\n")
    # ⛔ `nhiphan` da bo co chu y (cot `gia_rut_gon` nay nam tren moi dong). Cong
    #    khong duoc doi mot hien vat bai KHONG CON CO, neu khong no bao dong gia mai.
    #
    # Hai HO ket qua, va cac phep ap KHAC NHAU len chung:
    #   vps-*  quet A3 + heuristic. Co a3_* nen K1 (doi chung duong) va K5 (dem dong) ap duoc.
    #   cs-*   quet chinh sach hoc. KHONG co a3_* nen K1/K5 khong ap; bu lai co K9 (so tham so).
    # Cac phep bat bien chung (K2 thu tu DP, K3 gia rut gon, K7 troi nhi phan, K8 phan ra) ap
    # cho CA HAI, va do la diem cua viec gom chung vao mot cong.
    bang = {}
    for t_ in ("ofat", "mat2d"):
        bang["vps-" + t_] = doc("vps-%s.csv" % t_)
        bang["cs-" + t_] = doc("cs-%s.csv" % t_)
    for t, r in bang.items():
        if r is None:
            luu_y("chua co %s.csv" % t)
    co = {t: r for t, r in bang.items() if r}
    if not co:
        sys.exit("⛔ khong co tep ket qua nao")
    for t, r in co.items():
        print("  %-9s %7d dong, %3d che do, %2d chinh sach"
              % (t, len(r), len({x["che_do"] for x in r}), len({x["chinh_sach"] for x in r})))
    print()

    # ---- K1 doi chung duong ----
    for t in [k for k in co if k.startswith("vps-")]:
        theo = collections.defaultdict(dict)
        for x in co[t]:
            theo[(x["che_do"], x["vai"], x["seed"],
                  x.get("kinh_do", ""))][x["chinh_sach"]] = float(x["J"])
        lech = [(k, abs(v["a3_0_1"] - v["max-sinr"]))
                for k, v in theo.items() if "a3_0_1" in v and "max-sinr" in v]
        xau = [(k, d) for k, d in lech if d > 0]
        kiem(not xau and len(lech) > 0,
             "K1 %s: A3(hys=0,ttt=1) TRUNG KHIT max-SINR" % t,
             "%d/%d don vi, lech max %.2e" % (len(lech) - len(xau), len(lech),
                                              max([d for _, d in lech] or [0])))

    # ---- K2 thu tu DP ----
    for t, r in co.items():
        am = [x for x in r if float(x["gap_pct"]) < -1e-9]
        kiem(not am, "K2 %s: khong chinh sach nao tot hon DP day du" % t,
             "%d dong gap<0%s" % (len(am), (", te nhat %.3e" % min(float(x["gap_pct"]) for x in am))
                                  if am else ""))

    # ---- K3 gia rut gon nhi phan, nay doc tu CHINH cot gia_rut_gon ----
    for t_, r in co.items():
        if not r or "gia_rut_gon" not in r[0]:
            continue
        g = sorted(float(x["gia_rut_gon"]) for x in r)
        kiem(g[0] >= -1e-9, "K3 %s: DP nhi phan khong tot hon DP day du" % t_,
             "min %.6f%%, trung vi %.4f%%, max %.4f%%" % (g[0], g[len(g) // 2], g[-1]))

    # ---- K4 menh de suy bien ----
    if "vps-ofat" in co:
        hang = [x for x in co["vps-ofat"] if x["che_do"] == "fading_db=0.0" and x["vai"] == "train"]
        if not hang:
            luu_y("K4 khong tim duoc che do fading_db=0.0 trong ofat")
        else:
            a3 = collections.defaultdict(list)
            for x in hang:
                if x["chinh_sach"].startswith("a3_"):
                    a3[x["chinh_sach"]].append(float(x["gap_pct"]))
            tb = {k: sum(v) / len(v) for k, v in a3.items()}
            tot = min(tb, key=tb.get)
            kiem(tot == "a3_0_1",
                 "K4 menh de suy bien: o fading=0, A3 tot nhat la hys=0,ttt=1",
                 "tot nhat = %s (gap %.4f%%), a3_0_1 gap %.4f%%"
                 % (tot.replace("a3_", "hys="), tb[tot], tb.get("a3_0_1", float("nan"))))

    # ---- K5 du don vi ----
    # ⛔⛔ LAY TU BO CHAY, khong dem cung. Ban cu viet `2 + 15` tay, nen khi bo chay them
    #    mot chinh sach cam bien (`a3+a2a4`) thi cong bao THIEU DU LIEU tren mot tep DU.
    #    Do la bao dong gia, va bao dong gia tren mot cong dem dong la cach nhanh nhat de
    #    nguoi ta tap bo qua no. Nay danh sach chinh sach la MOT nguon: bo chay khai, cong doc.
    sys.path.insert(0, HERE)
    from chay_vps import DANH_SACH_CHINH_SACH
    n_cs = len(DANH_SACH_CHINH_SACH())
    for t in [k for k in co if k.startswith("vps-")]:
        # ⛔ Them `kinh_do`: sau khi lay mau lai hinh hoc, CUNG mot hat giong xuat hien o
        #    8 kinh do, nen bo no ra thi cong dem 270 don vi thay vi 2160 va bao thieu.
        dv = len({(x["che_do"], x["vai"], x["seed"], x.get("kinh_do", ""))
                  for x in co[t]})
        kiem(len(co[t]) == dv * n_cs, "K5 %s: so dong = so don vi x so chinh sach" % t,
             "%d dong vs %d x %d = %d" % (len(co[t]), dv, n_cs, dv * n_cs))

    # ---- K7 DP nhi phan phai TROI moi chinh sach NHI PHAN ----
    # ⛔ Va KHONG duoc doi dieu ay o max-SINR/max-elev: chung chon argmax tren MOI ve tinh
    #    nhin thay, tuc o khong gian DAY DU, nen chung CO THE tot hon DP nhi phan. Ap sai
    #    rang buoc len chung la bien mot phep kiem thanh mot bao dong gia.
    for t_, r in co.items():
        if not r or "khong_gian" not in r[0]:
            luu_y("K7 %s: chua co cot khong_gian, chay lai bo quet" % t_)
            continue
        nhi = [x for x in r if x["khong_gian"] == "binary"]
        xau = [x for x in nhi if float(x["gap_vs_bin"]) < -1e-9]
        kiem(not xau, "K7 %s: DP nhi phan troi moi chinh sach NHI PHAN" % t_,
             "%d/%d dong nhi phan vi pham" % (len(xau), len(nhi)))
        day = [x for x in r if x["khong_gian"] == "full"]
        tot_hon = [x for x in day if float(x["gap_vs_bin"]) < -1e-9]
        if tot_hon:
            luu_y("K7b %s: %d/%d dong khong-gian-DAY-DU tot hon DP nhi phan"
                  % (t_, len(tot_hon), len(day)),
                  "day la DUNG va dang chu y, khong phai loi")

    # ---- K8 phan ra phai CONG DUNG ----
    for t_, r in co.items():
        if not r or "gia_rut_gon" not in r[0]:
            continue
        lech = [abs(float(x["gap_pct"]) - float(x["gia_rut_gon"]) - float(x["gap_vs_bin"]))
                for x in r if x["khong_gian"] == "binary"]
        # gap(vs full) = gia_rut_gon + gap(vs bin) chi dung khi J_opt va J_bin cung dau; kiem bang so
        # ⛔ SIET tu 0,02 ve 2e-6. Muc 0,02 dat luc ba cot con khac mau so, nen no nuoi
        #    duoc mot lech 0,2257 diem ma khong bao. Nay ba cot cung mau so `|J_opt|` nen
        #    dang thuc DUNG theo cau tao, va nguong phai la bien LAM TRON (3 x 5e-7) chu
        #    khong phai mot so tron tay.
        kiem(max(lech or [0]) <= 2e-6,
             "K8 %s: phan ra gap = gia_rut_gon + gap_vs_bin" % t_,
             "lech max %.2e" % max(lech or [0]))

    # ---- K9 so tham so cua hai mo hinh phai KHONG DOI o moi dong ----
    # Bai hoi nghi ban "KAN ngang MLP voi it hon 3,2 lan tham so", nen hai con so 65 va 211 la
    # mot TUYEN BO cua bai. Mot dong lech la mot mo hinh khac da duoc huan luyen.
    for t_ in [k for k in co if k.startswith("cs-")]:
        sp = collections.defaultdict(set)
        for x in co[t_]:
            sp[x["chinh_sach"]].add(x["so_tham_so"])
        xau = {k: sorted(v) for k, v in sp.items() if len(v) != 1}
        kiem(not xau and sp.get("KAN") == {"65"} and sp.get("MLP") == {"211"},
             "K9 %s: so tham so khong doi, KAN 65 va MLP 211" % t_,
             json.dumps({k: sorted(v) for k, v in sp.items()}))
        if sp:
            ty = 211 / 65
            print("       ty le tham so MLP/KAN = %.2f lan (bai hoi nghi ban 3,2 lan)" % ty)

    # ---- K10 MOI MENH DE trong bai phai co MOT phep do do no ----
    # ⛔ Bai nay phat bieu hai menh de, va diem cua chung la chung DO DUOC chu khong duoc
    #    khang dinh. Neu mot menh de xuat hien trong ban thao ma khong co dong nao o day do
    #    no, thi bai dang ban mot dinh ly khong ai kiem. Nen cong doc THAN BAI va doi moi
    #    nhan menh de phai co mot phep do mang dung nhan ay.
    md = os.path.join(GOC, "paper", "sections")
    nhan_md = set()
    if os.path.isdir(md):
        for f in sorted(os.listdir(md)):
            if f.endswith(".tex"):
                s = io.open(os.path.join(md, f), encoding="utf-8").read()
                # ⛔ Quet CA `remark`. Ngay 16/10 phep phan ra bi ha tu Proposition xuong
                #    Remark (no la dong nhat thuc dai so, de o cho de thay nhat la dat sai
                #    trong tam). Cong chi quet `proposition` nen no mat dau tuyen bo VA van
                #    giu phep do, roi bao "co phep do cho menh de khong con trong bai" — mot
                #    canh bao dung ve hinh thuc nhung sai ve y: tuyen bo van o trong bai, chi
                #    doi nhan. Cong phai theo TUYEN BO, khong theo ten moi truong LaTeX.
                nhan_md |= set(re.findall(
                    r"\\begin\{(?:proposition|remark)\}[^\n]*\n\\label\{(prop:[^}]+)\}", s))
    do_duoc = {}
    nen = co.get("vps-ofat") or []
    if nen:
        b = [x for x in nen if x["khong_gian"] == "binary"]
        l = max(abs(float(x["gap_pct"]) - float(x["gia_rut_gon"]) - float(x["gap_vs_bin"]))
                for x in b)
        do_duoc["prop:decomp"] = ("lech lon nhat %.2e tren %d dong nhi phan" % (l, len(b)),
                                  l <= 2e-6)
        z = collections.defaultdict(list)
        for x in nen:
            if x["che_do"] == "fading_db=0.0":
                z[x["chinh_sach"]].append(float(x["J"]))
        if "a3_0_1" in z and "max-sinr" in z:
            d = max(abs(p0 - q0) for p0, q0 in zip(sorted(z["a3_0_1"]), sorted(z["max-sinr"])))
            do_duoc["prop:degen"] = ("lech |J| = %.2e o sigma=0" % d, d == 0.0)
    if not nhan_md:
        luu_y("K10 chua co menh de nao trong paper/sections")
    for n in sorted(nhan_md):
        if n not in do_duoc:
            kiem(False, "K10 menh de %s co phep do do no" % n, "KHONG co phep do nao")
        else:
            ct, ok = do_duoc[n]
            kiem(ok, "K10 menh de %s duoc phep do xac nhan" % n, ct)
    thua = sorted(set(do_duoc) - nhan_md)
    if thua:
        luu_y("K10b co phep do cho menh de khong con trong bai", ", ".join(thua))

    # ---- K6d MA DUNG CHUNG cho ra SO TRUNG KHIT, bang chung manh hon mot ma bam ----
    def _bang(ten, cot):
        # ⛔ Ghep phai theo (che do, hat giong, HINH HOC). Bo hinh hoc ra thi hai bo quet bi
        #    ghep nham giua cac kinh do khac nhau, va cong bao "19/135 hang lech" tren mot
        #    du lieu hoan toan dung. Mot phep doi chung ghep sai con te hon khong doi chung.
        r = doc(ten)
        return {(x["che_do"], x[cot], x.get("kinh_do", "")): (x["J_opt"], x["J_bin"])
                for x in r} if r else {}
    for a_, b_ in (("vps-ofat.csv", "cs-ofat.csv"), ("vps-mat2d.csv", "cs-mat2d.csv")):
        A, B = _bang(a_, "seed"), _bang(b_, "seed_test")
        gi = set(A) & set(B)
        if not gi:
            luu_y("K6d khong co phan giao giua %s va %s" % (a_, b_))
            continue
        lech = [k for k in gi if A[k] != B[k]]
        kiem(not lech,
             "K6d %s va %s: DP cho so trung khit tren phan giao" % (a_[:-4], b_[:-4]),
             "%d/%d hang lech" % (len(lech), len(gi)))

    # ---- K6 MOT may ----
    # ⛔⛔ PHAN BIEN NGOAI BAT DUNG CHO NAY, 11/10. Dong duoi day TUNG la `vps-*.xuat-xu.json`,
    #    nen cong NAY CHUA BAO GIO nhin cac tep `cs-*` — dung nhanh ma bai khai la nhay voi
    #    phien ban (chinh sach hoc lech 0,64% giua hai ban torch). Cong bao "MOT ban ma" trong
    #    khi co HAI, va bang xuat xu in ca hai ngay trong bai. Cong do DAI DIEN chu khong do
    #    dieu no khai bao ve: dung cai lop loi so tay da ghi, va no song sot vi khong ai tiem.
    xx = {}
    for p in sorted(glob.glob(os.path.join(RES, "*.csv.xuat-xu.json"))):
        xx[os.path.basename(p)] = json.load(io.open(p, encoding="utf-8"))
    if not xx:
        luu_y("K6 khong co tep xuat-xu nao")
    else:
        host = {v["host"] for v in xx.values()}
        goi = {json.dumps(v["goi"], sort_keys=True) for v in xx.values()}
        kiem(len(host) == 1, "K6a moi ket qua tu MOT host", ", ".join(sorted(host)))
        kiem(len(goi) == 1, "K6b moi ket qua tu MOT bo phien ban goi", "%d bo khac nhau" % len(goi))
        # ---- K6c: ma bam. Hai ho co the KHAC ma bam mot cach CHINH DANG neu manifest cua
        #      chung chi khac dung bo chay cua rieng ho. Nen phep kiem khong hoi "co mot ma
        #      bam khong" ma hoi "phan KHAC NHAU co dung la bo chay rieng khong".
        ds = {k: set(v.get("tep_ma") or []) for k, v in xx.items()}
        chung = set.intersection(*ds.values()) if ds else set()
        rieng = {k: sorted(v - chung) for k, v in ds.items() if v - chung}
        cho_phep = {"code/quet_chinh_sach.py"}
        la = {k: [z for z in v if z not in cho_phep] for k, v in rieng.items()}
        la = {k: v for k, v in la.items() if v}
        kiem(not la, "K6c manifest chi khac nhau o bo chay rieng cua tung ho",
             ("la: %s" % la) if la else ("%d tep chung, rieng: %s"
                                         % (len(chung), rieng or "khong")))
        bam = {v.get("ma_bam_code") for v in xx.values()}
        if None in bam:
            luu_y("K6c co tep xuat-xu THIEU ma_bam_code",
                  "chay lai bo chay de sinh lai: %d/%d tep"
                  % (sum(1 for v in xx.values() if not v.get("ma_bam_code")), len(xx)))
        # ⛔ KHONG con doi "MOT ban ma". Doi the la doi mot dieu KHONG DUNG: ma bam tinh tren
        #    TAP TEP, nen them dung bo chay cua rieng mot ho da doi ma bam du moi tep dung
        #    chung y nguyen. Bai tung khai "mot ma bam" nam lan, va do la mot cau SAI ma cong
        #    cu khong bat vi no chi nhin mot ho. Cai dang doi la K6c (manifest khac nhau o
        #    dung cho duoc phep) cong K6d (ma dung chung cho ra so trung khit), va ca hai
        #    deu MANH HON mot phep so chuoi bam.
        print("       ma bam theo ho: %s" % ", ".join(sorted(str(b)[:8] for b in bam)))

    print("\n=> %s (%d loi, %d chu y)"
          % ("DAT" if not loi else "CHUA DAT", len(loi), len(chuy)))
    for x in loi:
        print("   loi:", x)
    for x in chuy:
        print("   chu y:", x)
    return 1 if loi else 0


if __name__ == "__main__":
    sys.exit(main())
