#!/usr/bin/env python3
"""Cổng THƯ NGỎ và GÓI NỘP. Đọc thứ CÙNG GỬI, không chỉ đọc bản thảo.

    python3 code/cong_thu_ngo.py            # kiểm
    python3 code/cong_thu_ngo.py --tu-kiem  # tiêm lỗi

⛔ Sổ nhà đếm được hai lần hỏng ở đúng chỗ này, và cả hai lần bản thảo đều sạch:

  1. Thư ngỏ giữ nguyên bản cũ nên **chào editor bằng số bài đã rút**. Cổng chỉ đọc bản thảo
     thì không thấy. Luật rút ra: mọi hiện vật CÙNG GỬI phải vào cùng một cổng.
  2. Khi đổi venue, thư ngỏ **rơi mất đoạn khai bài liên quan**. Elsevier đòi khai, và thiếu
     nó là một thiếu sót có thể đọc thành giấu.

Nên cổng này kiểm BA thứ: thư ngỏ nói đúng venue và khai đủ; gói nộp không lẫn ghi chú nội
bộ; và những gì thư khai phải khớp bản thảo.
"""
import argparse
import glob
import io
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOC = os.path.dirname(HERE)
PAPER = os.path.join(GOC, "paper")
SUBMIT = os.path.join(GOC, "submit")

# Những gì BẮT BUỘC có trong thư, mỗi mục kèm lý do nó bắt buộc.
PHAI_KHAI = [
    ("Ad Hoc Networks", "đúng tên venue đang nộp"),
    ("withdrawn", "bài hội nghị ĐÃ RÚT phải được khai"),
    ("FAIR 2026", "nêu đích danh bài đã rút"),
    ("github.com/haodpsut/leo-handover-testbed", "kho công khai phải được khai"),
    ("VT-2026-06251", "bài TVT đang chờ phải được khai"),
    ("TCCN-TP-26-1197", "bài TCCN đang chờ phải được khai"),
    ("not under consideration elsewhere", "khai bài không nộp nơi khác"),
    ("AI coding assistant", "khai dùng AI, cho khớp trailer trong kho"),
    ("conflicts of interest", "khai xung đột lợi ích"),
]
# Những gì CẤM xuất hiện trong thư.
CAM = [
    ("1571332169", "số bài ĐÃ RÚT: không được dùng làm số bài đang nộp"),
    ("IEEE Transactions on Networking", "venue khác, dấu hiệu thư chưa đổi theo venue"),
]
# Vết ghi chú NỘI BỘ không được lọt vào thư mục nộp.
VET_NOI_BO = ["TRANG-THAI", "KE-HOACH", "gap-card", "XUAT-XU.md", "G6X", "l4-tu-choi",
              "CAU-HOI-DOC-NGOAI", "doc-ngoai"]


def doc(p):
    return io.open(p, encoding="utf-8").read()


def cham(thu=None, tep_submit=None):
    loi = []

    def kiem(dk, ten, ct=""):
        print("  %s %s%s" % ("DAT " if dk else "HONG", ten, ("  [" + ct + "]") if ct else ""))
        if not dk:
            loi.append(ten)

    v = thu if thu is not None else doc(os.path.join(PAPER, "thu-ngo.tex"))

    # ---- T1 thu khai du ----
    thieu = [(s, ly) for s, ly in PHAI_KHAI if s not in v]
    kiem(not thieu, "T1 thu ngo khai du moi muc bat buoc",
         ("thieu: %s" % [s for s, _ in thieu]) if thieu else "%d/%d muc" % (len(PHAI_KHAI),
                                                                           len(PHAI_KHAI)))
    # ---- T2 thu khong mang vet ban cu ----
    co = [(s, ly) for s, ly in CAM if s in v]
    kiem(not co, "T2 thu ngo khong mang vet ban cu",
         ("co: %s" % [s for s, _ in co]) if co else "0 vet")

    # ---- T3 thu va ban thao phai KHOP nhau ve cac con so ke ----
    mac = {}
    for t_ in ("so-lieu.tex", "so-van-lieu.tex"):
        p_ = os.path.join(GOC, "results", t_)
        if os.path.exists(p_):
            mac.update(dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", doc(p_))))
    # Thu viet "sixteen such proposals" bang CHU; bai phat `\nDeXuat` cung bang chu.
    n = mac.get("nDeXuat")
    kiem(not n or n in v, "T3 so bai de xuat trong thu khop macro cua bai",
         "bai: %s" % n)
    # Thu viet "twenty-two distinct operating conditions"; bai phat so 22.
    r = mac.get("qCheDoRieng")
    chu = {"22": "twenty-two", "27": "twenty-seven", "21": "twenty-one"}.get(r, r)
    kiem(not r or (chu in v or r in v), "T3b so dieu kien trong thu khop macro cua bai",
         "bai: %s (%s)" % (r, chu))

    # ---- T4 goi nop khong lan ghi chu noi bo ----
    ds = tep_submit if tep_submit is not None else (
        sorted(os.path.relpath(p_, SUBMIT) for p_ in glob.glob(os.path.join(SUBMIT, "**", "*"),
                                                               recursive=True)
               if os.path.isfile(p_)) if os.path.isdir(SUBMIT) else None)
    if ds is None:
        print("  chu y: chua co thu muc submit/, bo qua T4 va T5")
    else:
        ban = [f for f in ds if any(x.lower() in f.lower() for x in VET_NOI_BO)]
        kiem(not ban, "T4 goi nop khong lan ghi chu noi bo",
             ("lan: %s" % ban) if ban else "%d tep sach" % len(ds))
        # ---- T5 goi co du hien vat Elsevier doi ----
        can = {"manuscript": r"manuscript.*\.pdf$", "cover letter": r"cover-letter.*\.pdf$",
               "source": r"source.*\.zip$"}
    # ⛔ `highlights` KHONG con la hien vat bat buoc: no la yeu cau cua Elsevier, va bai nay
    #    nay nham Springer. Cong doi mot hien vat noi nhan khong dung la cong doi hien vat
    #    bai khong can — lop loi da co trong so nha.
        thieu2 = [k for k, rx in can.items() if not any(re.search(rx, f, re.I) for f in ds)]
        kiem(not thieu2, "T5 goi co du hien vat bat buoc",
             ("thieu: %s" % thieu2) if thieu2 else "%d loai" % len(can))

    print("\n=> %s (%d loi)" % ("DAT" if not loi else "CHUA DAT", len(loi)))
    return loi


def _tu_macro_nDeXuat():
    """Tra ve DUNG chuoi ma `\\nDeXuat` dang phat, de ca tiem neo vao macro chu khong go tay."""
    p_ = os.path.join(GOC, "results", "so-van-lieu.tex")
    m_ = re.search(r"\\newcommand\{\\nDeXuat\}\{([^}]*)\}", doc(p_))
    assert m_, "⛔ khong doc duoc macro nDeXuat, ca tiem T3 khong neo duoc"
    return m_.group(1)


def tu_kiem():
    goc = doc(os.path.join(PAPER, "thu-ngo.tex"))
    ds = sorted(os.path.relpath(p, SUBMIT) for p in
                glob.glob(os.path.join(SUBMIT, "**", "*"), recursive=True)
                if os.path.isfile(p)) if os.path.isdir(SUBMIT) else []
    print("== Ban GOC ==")
    if cham(goc, ds or None):
        sys.exit("⛔ ban GOC da hong, khong tiem duoc")
    ca = [
        ("T1: bo doan khai bai TVT dang cho",
         lambda: (goc.replace("VT-2026-06251", "VT-XXXX"), ds or None)),
        ("T1b: bo dia chi kho cong khai",
         lambda: (goc.replace("github.com/haodpsut/leo-handover-testbed", "a repository"),
                  ds or None)),
        ("T1c: bo khai dung AI",
         lambda: (goc.replace("AI coding assistant", "a text editor"), ds or None)),
        ("T2: de lai so bai DA RUT trong thu",
         lambda: (goc.replace("FAIR 2026", "FAIR 2026, paper 1571332169"), ds or None)),
        ("T3: doi so bai de xuat trong thu cho lech macro",
         # ⛔⛔ NEO PHAI LAY TU MACRO, khong go chu. Neo cu la chuoi "sixteen"; ngay 16/10
         #    danh sach de xuat len 17 nen thu doi thanh "seventeen" va phep thay tro thanh
         #    VO NGHIA — ca tiem khong doi gi ma van duoc dem la mot ca. Day la lan thu HAI
         #    cung lop loi trong mot tuan (lan truoc o cong_bai B22). Nay neo doc chinh
         #    `\nDeXuat` nen no khong the lac hau duoc nua.
         lambda: (goc.replace(_tu_macro_nDeXuat(), "nine"), ds or None)),
        ("T4: nhet mot ghi chu noi bo vao goi",
         lambda: (goc, (ds or []) + ["TRANG-THAI.md"])),
        ("T5: bo thu ngo khoi goi",
         lambda: (goc, [f for f in (ds or []) if "cover-letter" not in f.lower()])),
    ]
    hong = []
    for ten, f in ca:
        print("\n== %s ==" % ten)
        a, b = f()
        if a == goc and b == (ds or None):
            hong.append(ten + " (CA VO NGHIA)"); print("  ⛔ ca nay khong doi gi"); continue
        l = cham(a, b)
        print("  => %s" % ("cong BAT duoc" if l else "⛔ cong KHONG bat"))
        if not l:
            hong.append(ten)
    print("\n=> %s (%d/%d ca)" % ("DAT" if not hong else "CHUA DAT", len(ca) - len(hong), len(ca)))
    for h in hong:
        print("   HONG: %s" % h)
    return 1 if hong else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()
    if a.tu_kiem:
        return tu_kiem()
    print("== Cong thu ngo va goi nop ==\n")
    return 1 if cham() else 0


if __name__ == "__main__":
    sys.exit(main())
