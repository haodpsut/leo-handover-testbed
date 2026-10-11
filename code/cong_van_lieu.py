#!/usr/bin/env python3
"""Cổng VĂN LIỆU: những tuyên bố ĐẾM ĐƯỢC về văn liệu phải do máy đếm.

    python3 code/cong_van_lieu.py            # kiểm, và sinh results/so-van-lieu.tex
    python3 code/cong_van_lieu.py --tu-kiem  # tiêm lỗi để chứng minh cổng có bắt

⛔ Sinh từ một lỗi thật ở chính bài này, 11/10/2026: §Introduction viết *"twelve distinct papers
proposing a new LEO handover mechanism"* trong khi danh sách `\\citep` ngay cạnh có **13** khoá,
rồi sau đó thêm ba bài nữa thành 16 mà câu văn vẫn nói "twelve". Con số ấy là một tuyên bố kiểm
được, nên nó không được phép là một chữ gõ tay: cổng này đếm khoá giữa hai mốc trong
`02-related.tex`, phát ra macro `\\nDeXuat`, và bài dùng macro. Cùng lớp lỗi với
"bảng/hình gõ tay thì chạy lại thí nghiệm không cập nhật", chỉ là ở văn liệu thay vì ở số đo.

Ba phép kiểm:
  V1  mọi khoá được trích đều có trong refs.bib, và mọi mục refs.bib đều được trích
  V2  danh sách đề xuất giữa hai mốc không rỗng, không trùng khoá, và không lẫn survey
  V3  mọi mục refs.bib có đúng một mã tự giải được (doi, hoặc arXiv trong journal, hoặc
      số tài liệu chuẩn), vì luật nhà cấm đưa vào bài một mã không tự giải được
"""
import argparse
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOC = os.path.dirname(HERE)
PAPER = os.path.join(GOC, "paper")
RES = os.path.join(GOC, "results")
BAT, KET = "% MOC-DE-XUAT-BAT-DAU", "% MOC-DE-XUAT-KET-THUC"
# Khoá KHÔNG được nằm trong danh sách đề xuất, dù chúng cũng là bài LEO:
KHONG_PHAI_DE_XUAT = {"park2021trends", "badini2026",      # survey
                      "demir2022", "kodheli2024", "juan2022",  # đo lường cơ chế chuẩn
                      "benzaghta2025",                     # chỉnh tham số, không co che moi
                      "namyar2022", "karimi2024xplain", "hossain2026satlock", "mazilu2025cc",
                      "michel2022starlink", "laniewski2024", "jang2025geo", "zhao2026aviation",
                      "liu2024kan", "vallado2006",
                      "3gpp38300", "3gpp38331", "3gpp38821"}


def doc(p):
    return io.open(p, encoding="utf-8").read()


def khoa_trich(van):
    ra = []
    for m in re.findall(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", van):
        ra += [k.strip() for k in m.split(",") if k.strip()]
    return ra


def cham(tex_related=None, bib=None, cac_tep=None):
    """Trả về (loi, so_de_xuat). Tham số để phép tiêm thay nội dung mà không ghi đĩa."""
    loi = []

    def kiem(dk, ten, ct=""):
        print("  %s %s%s" % ("DAT " if dk else "HONG", ten, ("  [" + ct + "]") if ct else ""))
        if not dk:
            loi.append(ten)

    van_rel = tex_related if tex_related is not None else doc(
        os.path.join(PAPER, "sections", "02-related.tex"))
    van_bib = bib if bib is not None else doc(os.path.join(PAPER, "refs.bib"))
    tep = cac_tep if cac_tep is not None else {
        p: doc(p) for p in glob.glob(os.path.join(PAPER, "sections", "*.tex"))
        + [os.path.join(PAPER, "main.tex")]}

    muc = set(re.findall(r"@\w+\{([^,]+),", van_bib))
    dung = set()
    for p, v in tep.items():
        dung |= set(khoa_trich(v if os.path.basename(p) != "02-related.tex" else van_rel))

    # ---- V1 ----
    kiem(not (dung - muc), "V1a moi khoa duoc trich deu co trong refs.bib",
         "thieu: %s" % sorted(dung - muc) if dung - muc else "0 thieu")
    kiem(not (muc - dung), "V1b moi muc refs.bib deu duoc trich",
         "treo: %s" % sorted(muc - dung) if muc - dung else "0 treo")

    # ---- V2 ----
    if BAT not in van_rel or KET not in van_rel:
        kiem(False, "V2a co hai moc danh sach de xuat trong 02-related.tex")
        return loi, 0
    giua = van_rel.split(BAT, 1)[1].split(KET, 1)[0]
    dx = khoa_trich(giua)
    kiem(len(dx) >= 5, "V2a danh sach de xuat khong rong", "%d khoa" % len(dx))
    trung = {k for k in dx if dx.count(k) > 1}
    kiem(not trung, "V2b khong khoa nao lap trong danh sach de xuat",
         "lap: %s" % sorted(trung) if trung else "0 lap")
    lan = sorted(set(dx) & KHONG_PHAI_DE_XUAT)
    kiem(not lan, "V2c khong lan survey/do-luong/chuan vao danh sach de xuat",
         "lan: %s" % lan if lan else "0 lan")

    # ---- V3 ----
    thieu_ma = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", van_bib, re.S):
        loai, k, than = m.group(1), m.group(2), m.group(3)
        co = ("doi" in than.lower() and re.search(r"doi\s*=\s*\{\S", than)) \
            or re.search(r"arXiv:\d{4}\.\d{4,5}", than) \
            or re.search(r"number\s*=\s*\{(TS|TR) \d", than)
        if not co:
            thieu_ma.append(k)
    kiem(not thieu_ma, "V3 moi muc refs.bib co mot ma tu giai duoc",
         "khong ma: %s" % thieu_ma if thieu_ma else "%d muc deu co" % len(muc))
    return loi, len(set(dx))


def tu_kiem():
    """Mỗi ca phải làm cổng CHUYỂN TỪ ĐẠT SANG HỎNG. Ca nào không đổi gì là ca vô nghĩa."""
    rel = doc(os.path.join(PAPER, "sections", "02-related.tex"))
    bib = doc(os.path.join(PAPER, "refs.bib"))
    tep = {p: doc(p) for p in glob.glob(os.path.join(PAPER, "sections", "*.tex"))
           + [os.path.join(PAPER, "main.tex")]}
    print("== Goc ==")
    goc, n = cham(rel, bib, tep)
    if goc:
        sys.exit("⛔ ban GOC da hong, khong tiem duoc: %s" % goc)
    print("  (goc DAT, %d bai de xuat)\n" % n)

    ca = [
        ("tiem khoa khong co trong bib",
         lambda: (rel.replace("\\citep{zhao2026twc}", "\\citep{bai-khong-ton-tai}"), bib, tep)),
        ("tiem lan SURVEY vao danh sach de xuat",
         lambda: (rel.replace("variants~\\citep{yan2023,huang2026}",
                              "variants~\\citep{yan2023,huang2026,park2021trends}"), bib, tep)),
        ("tiem khoa LAP trong danh sach de xuat",
         lambda: (rel.replace("\\citep{chu2025}", "\\citep{chu2025,zhao2026twc}"), bib, tep)),
        ("tiem muc bib KHONG co ma tu giai duoc",
         lambda: (rel, bib + "\n@article{bai-khong-ma,\n  author  = {A B},\n"
                  "  title   = {Khong co ma nao},\n  journal = {Tap chi X},\n"
                  "  year    = {2026}\n}\n",
                  {p: (v + "\n\\citep{bai-khong-ma}") if p.endswith("main.tex") else v
                   for p, v in tep.items()})),
        ("xoa MOC danh sach de xuat",
         lambda: (rel.replace(BAT, "% da xoa moc"), bib, tep)),
    ]
    hong = []
    for ten, f in ca:
        r, b, s = f()
        doi = (r != rel) or (b != bib) or (s != tep)
        if not doi:
            hong.append(ten + " (CA VO NGHIA: khong doi gi)")
            print("== %s ==\n  ⛔ ca nay khong doi gi\n" % ten)
            continue
        print("== %s ==" % ten)
        l, _ = cham(r, b, s)
        print("  => %s\n" % ("cong BAT duoc" if l else "⛔ cong KHONG bat"))
        if not l:
            hong.append(ten)
    print("=> %s (%d/%d ca)" % ("DAT" if not hong else "CHUA DAT",
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
    print("== Cong van lieu ==\n")
    loi, n = cham()
    p = os.path.join(RES, "so-van-lieu.tex")
    if loi:
        print("\n=> CHUA DAT (%d loi). KHONG ghi %s." % (len(loi), os.path.basename(p)))
        return 1
    io.open(p, "w", encoding="utf-8").write(
        "%% SINH TU DONG boi code/cong_van_lieu.py. KHONG sua tay.\n"
        "\\newcommand{\\nDeXuat}{%s}\n" % {12: "twelve", 13: "thirteen", 14: "fourteen",
                                           15: "fifteen", 16: "sixteen", 17: "seventeen",
                                           18: "eighteen"}.get(n, str(n)))
    print("\n=> DAT. %d bai de xuat -> %s" % (n, os.path.relpath(p, GOC)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
