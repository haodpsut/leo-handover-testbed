#!/usr/bin/env python3
"""Bộ chạy SONG SONG trên sol1, có khai xuất xứ và chạy lại được.

    python3 code/chay_vps.py --viec ofat      --nhan 24
    python3 code/chay_vps.py --viec mat2d     --nhan 24
    python3 code/chay_vps.py --viec chinhsach --nhan 16
    python3 code/chay_vps.py --viec nhiphan   --nhan 24
    python3 code/chay_vps.py --tu-kiem

⛔ NĂM LUẬT, mỗi luật từ một ca thật:

1. **`OMP_NUM_THREADS=1` cho MỖI tiến trình con, đặt TRƯỚC khi import numpy/torch.** sol1 có
   6.113 người đăng nhập; một tiến trình torch mở 80 luồng thì cả máy đứng và không phải máy của
   mình. Đặt sau khi import là vô tác dụng, nên nó nằm ở đầu tệp.
2. **Chừa nhân.** Mặc định 24 trên 80, không phải 80. Máy dùng chung, và tải nền đã ~4.
3. **GHI TỪNG DÒNG, không gom cuối.** Bị giết giữa lúc chạy là chuyện thường trên máy dùng chung;
   gom cuối thì mất trắng. Mỗi dòng xong là `flush` ngay, và bộ chạy **bỏ qua** đơn vị việc đã có
   trong tệp ra, nên chạy lại là tiếp tục chứ không phải làm lại.
4. **KHAI MÁY VÀ MÔI TRƯỜNG CÙNG LÚC VỚI CON SỐ.** Mỗi tệp ra kèm một tệp `.xuat-xu.json` ghi
   hostname, python, phiên bản numpy/torch/skyfield/sgp4, commit của kho gốc, và giờ UTC. Ca thật:
   lệch host đo được ở bài này là **0,64% trên Σ|w| của MLP** giữa torch 2.8.0 và 2.14.1.
5. ⛔ **KHÔNG đặt token GitHub lên máy này.** Chạy xong thì `rsync` kết quả về Mac, commit và đẩy
   từ Mac.
"""
import argparse
import csv
import datetime
import io
import itertools
import json
import os
import platform
import socket
import sys
import time

# ⛔ Phải đặt TRƯỚC mọi import numpy/torch, nếu không thì vô tác dụng.
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"

import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOC = os.path.dirname(HERE)
# ⛔⛤ CAY LAM VIEC PHAI THANG CAY TRONG KHO. Hai ban cua cung mot tep ton tai: `code/`
#    (dang sua) va `repo-goc/code/` (hien vat da cong bo). Mot vong `sys.path.insert(0, ...)`
#    thi cai CHEN SAU dung TRUOC, nen tuy thu tu liet ke ma ban nao che ban nao — va mot ban
#    CU trong kho da che ban dang sua, bao loi "khong import duoc ham vua them". Dat tuong
#    minh: kho di truoc trong danh sach, roi cay lam viec duoc nhet len dau SAU CUNG.
for _p in (os.path.join(GOC, "repo-goc", "code"), os.path.join(GOC, "code")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
if os.path.isdir(os.path.join(GOC, "code")):
    sys.path.remove(os.path.join(GOC, "code"))
    sys.path.insert(0, os.path.join(GOC, "code"))
import handover as H                                       # noqa: E402
from camera_ready_extra import a3_ttt                      # noqa: E402
from quet_che_do import NEO, HYS, TTT, gap, ten_che_do     # noqa: E402

RES = os.path.join(GOC, "results")

# Hai trục có hiệu ứng LỚN NHẤT, đo được ở lượt thăm dò: vĩ độ nở 4,4 lần, giá chuyển giao nở
# 3,5 lần. Nên mặt 2D đi theo CHÚNG, không theo mật độ × pha đinh như bản kế hoạch đầu.
VI_DO_2D = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0]
GIA_HO_2D = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _doc_omp(_):
    """Phai la ham CAP MODULE: `os.environ.get` khong pickle duoc nen `pool.map` len no se no."""
    return os.environ.get("OMP_NUM_THREADS")


def xuat_xu():
    """Khai máy và môi trường. Đi KÈM mỗi tệp số, không để rời."""
    pv = {}
    for m in ("numpy", "torch", "skyfield", "sgp4", "sklearn", "matplotlib"):
        try:
            pv[m] = getattr(__import__(m), "__version__", "?")
        except ImportError:
            pv[m] = None
    commit = None
    for d in (os.path.join(GOC, "repo-goc"), GOC):
        g = os.path.join(d, ".git")
        if os.path.isdir(g):
            try:
                commit = io.open(os.path.join(g, "HEAD"), encoding="utf-8").read().strip()
                if commit.startswith("ref:"):
                    commit = io.open(os.path.join(g, commit[5:]), encoding="utf-8").read().strip()
            except OSError:
                pass
            break
    # ⛔ `commit_kho_goc` ra `null` trên sol1, vì khi đẩy mã lên tôi loại `.git`, nên trên đó
    #    không có gì để đọc. Một tệp xuất xứ khai `null` ở đúng trường định danh mã là một tệp
    #    xuất xứ không dùng được. Nên băm CHÍNH CÁC TỆP MÃ: chặt hơn commit id, vì nó bắt cả
    #    sửa tay chưa commit, và nó hoạt động ở mọi máy không cần git.
    import hashlib
    bam = hashlib.sha256()
    tep_ma = []
    for d in (os.path.join(GOC, "code"), os.path.join(GOC, "repo-goc", "code")):
        if os.path.isdir(d):
            for ten in sorted(os.listdir(d)):
                if ten.endswith(".py"):
                    q = os.path.join(d, ten)
                    bam.update(ten.encode())
                    bam.update(io.open(q, "rb").read())
                    tep_ma.append(os.path.relpath(q, GOC))
    return {"host": socket.gethostname(), "may": platform.platform(),
            "python": sys.version.split()[0], "goi": pv, "commit_kho_goc": commit,
            "ma_bam_code": bam.hexdigest()[:16], "so_tep_ma": len(tep_ma),
            "tep_ma": tep_ma,
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "nhan_cpu": os.cpu_count()}


# ----------------------------- đơn vị việc -----------------------------------

# ⛔⛔ HAI KHONG GIAN HANH DONG, va day la mot SUA DOI CONG THUC chu khong phai chi tiet:
#
#   A3 chon trong {giu, chuyen sang ung vien manh nhat}  => NHI PHAN
#   chinh sach hoc cung vay (xem `handover_binary.rollout`) => NHI PHAN
#   max-SINR va max-elev chon argmax tren MOI ve tinh nhin thay => DAY DU
#   DP day du khong bi rang buoc => DAY DU
#
# Nen so A3 voi DP DAY DU la so mot co che nhi phan voi mot chan NO KHONG VOI TOI DUOC. Do
# duoc: phan "gia cua phep rut gon nhi phan" chiem TRUNG VI 13,4% cua khoang cach, va o
# `fading_db=0` thi chiem 53,5%, tuc hon NUA khoang cach o do khong phai loi cua A3.
#
# ⇒ Moi dong phai mang CA HAI goc do, de bai phan ra duoc:
#     gap(chinh sach nhi phan vs DP day du) = gia_rut_gon + gap(chinh sach vs DP nhi phan)
# DP nhi phan ton 0,005 s so voi 0,193 s cua DP day du, tuc 2,6%, nen gop vao la mien phi.
KHONG_GIAN = {"max-sinr": "full", "max-elev": "full"}        # con lai (a3_*) la "binary"


def _mot_diem_a3(kw, seed, lon=0.0):
    """DP đầy đủ + DP nhị phân + heuristic + toàn lưới A3, cho MỘT chế độ và MỘT thực thể.

    `lon` xoay KINH ĐỘ trạm, tức lấy một mẫu hình học khác của cùng chế độ. Vĩ độ do chế
    độ quyết định và không bị đụng tới.
    """
    import handover_binary as HB
    p = dict(NEO); p.update(kw)
    lat = p["station"][0]
    p["station"] = (lat, lon)
    sc = H.build_scenario(seed=seed, **p)
    J_opt, _ = H.dp_optimum(sc)
    _V, _ex, serv_bin = HB.solve_dp(sc)
    J_bin = H.evaluate(serv_bin, sc)["J"]
    assert J_bin >= J_opt - 1e-9, "DP nhi phan tot hon DP day du: %.9f < %.9f" % (J_bin, J_opt)
    r = {"J_opt": J_opt, "J_bin": J_bin,
         "vis": sum(len(v) for v in sc.visible) / sc.S, "N": sc.N, "S": sc.S}
    r["max-sinr"] = H.evaluate(H.greedy_max_sinr(sc), sc)
    r["max-elev"] = H.evaluate(H.greedy_max_elev(sc), sc)
    for h, t in itertools.product(HYS, TTT):
        r["a3_%g_%d" % (h, t)] = H.evaluate(a3_ttt(sc, hys_db=h, ttt=t), sc)
    return r


def viec_a3(dv):
    """dv = (ten_viec, nhan_che_do, kw, seed, vai, kinh_do). Trả về các dòng CSV."""
    ten, nhan, kw, seed, vai, lon = dv
    r = _mot_diem_a3(kw, seed, lon)
    hang = []
    for k in ["max-sinr", "max-elev"] + ["a3_%g_%d" % (h, t)
                                         for h, t in itertools.product(HYS, TTT)]:
        e = r[k]
        hang.append(dict(viec=ten, che_do=nhan, vai=vai, seed=seed, kinh_do=lon,
                         chinh_sach=k,
                         khong_gian=KHONG_GIAN.get(k, "binary"),
                         gap_pct=round(gap(e["J"], r["J_opt"]), 6),
                         # ⛔ CUNG MOT MAU SO |J_opt| cho ca ba khoang cach. Ban dau toi dung
                         #    `gap(J, J_bin)` tuc mau so |J_bin|, va cong K8 bat duoc ngay:
                         #    phep phan ra lech toi 0,2257 diem. Ba khoang cach chi CONG DUOC
                         #    khi chung cung mot goc chuan hoa, va goc dung la toi uu KHONG bi
                         #    rang buoc, vi do la thu ca bai do theo.
                         gap_vs_bin=round(100.0 * (e["J"] - r["J_bin"]) / abs(r["J_opt"]), 6),
                         gia_rut_gon=round(gap(r["J_bin"], r["J_opt"]), 6),
                         J=round(e["J"], 6), J_opt=round(r["J_opt"], 6),
                         J_bin=round(r["J_bin"], 6),
                         pingpong=e["pingpong"], handovers=e["handovers"],
                         outage=e["outage"], vis_per_slot=round(r["vis"], 4),
                         N=r["N"], S=r["S"]))
    return hang




# ----------------------------- danh sách việc --------------------------------

# ⛔⛔ HINH HOC PHAI DUOC LAY MAU LAI. Hoi dong G6 (12/10) bat duoc: `build_scenario` chi
#    gieo PHA DINH; thoi diem quan sat va kinh do tram la hang so viet cung, nen
#    `max|el(seed 0) - el(seed 100)| = 0`. Voi mot truc HINH HOC nhu vi do hay mat do thi
#    toan bo bo quet cu la **n = 1**, va khong co thanh sai so nao ton tai. Phan bien lay
#    mau lai bang cach xoay kinh do: o 70 do khoang cach trai tu 2,24% den 5,53%, tuc con
#    so cu nam gan DAY phan bo cua chinh no.
#
#    Nay mot THUC THE = mot cap (hinh hoc, pha dinh). Hinh hoc lay bang cach xoay kinh do
#    tram 360/G do: voi vo Walker, xoay kinh do doi hoan toan ve tinh nao o tren dau, nen
#    day la mot phep lay mau hinh hoc that chu khong phai doi nhan.
#    ⛔ Khong dong vao `repo-goc/code/handover.py`: `build_scenario` da nhan `station=(lat,lon)`,
#    nen phep lay mau nay khong sua gi trong bo mo phong da cong bo.
#
#    NGAN SACH GIU NGUYEN: truoc la 40 hat giong pha dinh tren 1 hinh hoc; nay la
#    G=8 hinh hoc x 5 hat giong pha dinh = 40 thuc the moi vai. Cung so don vi, nhung
#    phuong sai do duoc nay gom ca phan hinh hoc, la phan tung bi giau hoan toan.
G_HINH = 8


def _bo_thuc_the(n_train, n_test):
    """Tra ve [(vai, seed, kinh_do)], G hinh hoc x (n/G) hat giong pha dinh moi vai."""
    f_tr = max(1, n_train // G_HINH)
    f_te = max(1, n_test // G_HINH)
    tr = [("train", s, 360.0 * g / G_HINH)
          for g in range(G_HINH) for s in range(f_tr)]
    te = [("test", s, 360.0 * g / G_HINH)
          for g in range(G_HINH) for s in range(100, 100 + f_te)]
    return tr, te


def don_vi(viec, n_train, n_test):
    tr, te = _bo_thuc_the(n_train, n_test)
    from quet_che_do import TRUC
    if viec == "ofat":
        diem = [("neo", {})] + [(ten_che_do(kw), kw) for t in TRUC for kw in TRUC[t]]
        # bỏ trùng: điểm neo xuất hiện lại trong vài trục
        thay, ds = set(), []
        for nhan, kw in diem:
            if nhan in thay:
                continue
            thay.add(nhan); ds.append((nhan, kw))
        return [("ofat", nhan, kw, s, vai, lon) for nhan, kw in ds for vai, s, lon in tr + te]
    if viec == "mat2d":
        return [("mat2d", "vido=%g+lam3=%g" % (v, c), dict(station=(v, 0.0), lam3=c), s, vai, lon)
                for v in VI_DO_2D for c in GIA_HO_2D for vai, s, lon in tr + te]
    raise SystemExit("⛔ viec khong biet: %s" % viec)


# ⛔ `nhiphan` DA BO: cot `gia_rut_gon` nay nam tren MOI dong cua ofat/mat2d, nen mot
#    viec rieng chi de lai mot tep ket qua cu khong ai cap nhat. Cong K6c bat duoc
#    dung chuyen do: hai ban ma khac nhau trong cung mot thu muc results.
HAM = {"ofat": viec_a3, "mat2d": viec_a3}


def da_lam(ra):
    """Đơn vị việc đã có trong tệp ra. Để chạy lại là TIẾP TỤC, không làm lại."""
    if not os.path.exists(ra):
        return set()
    with io.open(ra, encoding="utf-8") as f:
        # ⛔ KHOA PHAI GOM CA KINH DO. Mot thuc the nay la (che do, vai, hat giong, hinh
        #    hoc); bo kinh do ra thi chay tiep se coi 8 mau hinh hoc la MOT va bo sot 7.
        return {(r["che_do"], r["vai"], r["seed"], r.get("kinh_do", ""))
                for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viec", default="ofat", choices=list(HAM))
    ap.add_argument("--nhan", type=int, default=24, help="so tien trinh; MAY DUNG CHUNG, dung lay het")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()

    if a.tu_kiem:
        loi = []
        # 1. OMP phai bang 1 trong tien trinh CON, khong chi trong tien trinh cha
        with mp.Pool(2) as pool:
            v = pool.map(_doc_omp, [0, 1])
        ok = all(x == "1" for x in v)
        print("  %s tu-kiem 1: OMP_NUM_THREADS=1 trong tien trinh con  [%s]"
              % ("DAT " if ok else "HONG", v))
        loi += [] if ok else ["omp"]
        # 2. chay lai phai TIEP TUC: don vi da lam bi bo qua
        ra = os.path.join(RES, "_tu-kiem.csv")
        io.open(ra, "w", encoding="utf-8").write(
            "che_do,vai,seed,kinh_do\nneo,test,100,45.0\n")
        d = da_lam(ra)
        ok2 = ("neo", "test", "100", "45.0") in d
        print("  %s tu-kiem 2: doc duoc don vi da lam tu tep ra  [%d don vi]"
              % ("DAT " if ok2 else "HONG", len(d)))
        loi += [] if ok2 else ["tiep-tuc"]
        # 2b. khoa phai KHOP VONG KHU HOI: cach bo chay sinh khoa == cach CSV ghi ra
        dv0 = don_vi("ofat", 8, 8)[0]
        khoa_sinh = (dv0[1], dv0[4], str(dv0[3]), str(dv0[5]))
        io.open(ra, "w", encoding="utf-8").write(
            "che_do,vai,seed,kinh_do\n%s,%s,%s,%s\n"
            % (dv0[1], dv0[4], dv0[3], dv0[5]))
        ok2b = khoa_sinh in da_lam(ra)
        os.remove(ra)
        print("  %s tu-kiem 2b: khoa tiep-tuc khop vong khu hoi  [%s]"
              % ("DAT " if ok2b else "HONG", khoa_sinh))
        loi += [] if ok2b else ["khoa-khu-hoi"]
        # 3. xuat xu phai co host va phien ban
        x = xuat_xu()
        ok3 = bool(x["host"]) and x["goi"].get("numpy")
        print("  %s tu-kiem 3: khai duoc host va phien ban goi  [%s, numpy %s]"
              % ("DAT " if ok3 else "HONG", x["host"], x["goi"].get("numpy")))
        loi += [] if ok3 else ["xuat-xu"]
        print("\n=> %s (%d loi)" % ("DAT" if not loi else "CHUA DAT", len(loi)))
        return 1 if loi else 0

    os.makedirs(RES, exist_ok=True)
    ra = os.path.join(RES, "vps-%s.csv" % a.viec)
    dv = don_vi(a.viec, a.n_train, a.n_test)
    xong = da_lam(ra)
    # ⛔ Dung `str()` chu khong `"%g"`: CSV ghi `45.0` con `"%g" % 45.0` cho `45`, nen khoa
    #    khong bao gio khop va chay tiep se lam lai TAT CA. An toan nhung vo ich, va kieu
    #    sai lech nay chi lo ra khi so hai chuoi that.
    con = [d for d in dv if (d[1], d[4], str(d[3]), str(d[5])) not in xong]
    print("== chay_vps: viec=%s  %d don vi, %d da lam, %d con lai, %d nhan =="
          % (a.viec, len(dv), len(dv) - len(con), len(con), a.nhan))
    # ⛔ GHI XUAT XU TRUOC moi loi ra som. Ban dau no nam SAU `if not con: return`, nen mot
    #    lan chay lai khong con viec gi se KHONG ghi lai xuat xu, va cong K6c bao thieu
    #    `ma_bam_code` tren dung ba tep khong co cach nao sinh lai ngoai viec xoa ket qua.
    #    Xuat xu phai luon la hien tai, bat ke co viec hay khong.
    io.open(ra + ".xuat-xu.json", "w", encoding="utf-8").write(
        json.dumps(xuat_xu(), indent=1, ensure_ascii=False))

    if not con:
        print("  khong con gi de lam (xuat xu da ghi lai).")
        return 0

    moi = not os.path.exists(ra)
    f = io.open(ra, "a", newline="", encoding="utf-8")
    w, t0, n = None, time.time(), 0
    with mp.Pool(a.nhan) as pool:
        for hang in pool.imap_unordered(HAM[a.viec], con, chunksize=1):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(hang[0].keys()))
                if moi:
                    w.writeheader()
            w.writerows(hang)
            f.flush()                       # ⛔ flush TUNG DONG, xem luat 3
            n += 1
            if n % 50 == 0 or n == len(con):
                d = time.time() - t0
                print("  %5d/%d  %.1f s  (%.2f don vi/s, con ~%.0f s)"
                      % (n, len(con), d, n / d, (len(con) - n) / (n / d)), flush=True)
    f.close()
    print("  -> %s" % os.path.relpath(ra, GOC))
    print("  -> %s" % os.path.relpath(ra + ".xuat-xu.json", GOC))
    return 0


if __name__ == "__main__":
    sys.exit(main())
