"""Optional PDF export of the research manuscript (reportlab + Pillow)."""
import re
from html import escape
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ALIASES = {"asset": "Tài sản", "dataset": "Nguồn", "first_date": "Ngày đầu", "last_date": "Ngày cuối",
           "expected_days": "Ngày lịch", "valid_days": "Ngày hợp lệ", "coverage": "Độ phủ",
           "invalid_snapshots": "Snapshot lỗi", "missing_days": "Ngày thiếu", "cohort": "Mẫu",
           "split": "Phần mẫu", "calendar_days": "Ngày lịch", "complete_feature_days": "Đủ biến",
           "excluded_days": "Loại", "mean_log_return": "Mean r", "annual_volatility": "SD năm",
           "skewness": "Skew", "excess_kurtosis": "Kurt dư", "heavy_tail": "t5", "repetitions": "Số lặp",
           "transition_rmse": "RMSE P", "transition_rmse_mcse": "MCSE P",
           "volatility_relative_rmse": "RMSE σ", "passes_project_precision_targets": "Đạt",
           "validation_log_score": "Score val", "train_bic": "BIC train", "n_parameters": "Tham số",
           "min_posterior_days": "Min ngày", "limited_states": "State ít",
           "regime": "Trạng thái", "mean_return_daily": "Mean r/ngày", "volatility_annual": "SD năm",
           "occupancy": "Tỷ phần", "persistence": "p_ii", "expected_duration_days": "Duration ngày",
           "score_group": "Nhóm score", "test_days": "Ngày test", "mean_log_score": "Joint/return score",
           "mean_return_log_score": "Return score", "nonconverged_refits": "Không hội tụ",
           "state_changes": "Đổi state", "n_common": "Ngày chung", "mean_difference": "Chênh lệch",
           "ci_low": "CI thấp", "ci_high": "CI cao", "strategy": "Chiến lược", "cagr": "CAGR",
           "volatility": "SD năm", "sharpe": "Sharpe", "max_drawdown": "Max DD",
           "turnover": "Turnover", "ending_value": "Giá trị cuối", "left": "Mô hình 1", "right": "Mô hình 2",
           "raw_high_vol": "Mã σ max", "parameter": "Tham số", "estimate": "Ước lượng",
           "std_error": "SE", "volatility_low": "SD CI thấp", "volatility_high": "SD CI cao",
           "duration_low": "Duration CI thấp", "duration_high": "Duration CI cao"}

def inline(text):
    text = escape(text.replace("–", "-").replace("—", "-").replace("−", "-"))
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#3566A8">\1</link>', text)
    return text


def render():
    font_dir = Path("C:/Windows/Fonts")
    filenames = ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"]
    if not (font_dir / filenames[0]).exists():
        from matplotlib import get_data_path
        font_dir = Path(get_data_path()) / "fonts/ttf"
        filenames = ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
                     "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf"]
    for name, filename in zip(("Research", "Research-Bold", "Research-Italic", "Research-BoldItalic"), filenames):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily("Research", normal="Research", bold="Research-Bold",
                                  italic="Research-Italic", boldItalic="Research-BoldItalic")
    body = ParagraphStyle("ResearchBody", fontName="Research", fontSize=9.6, leading=14.2,
                          alignment=TA_JUSTIFY, spaceAfter=7, splitLongWords=True, allowWidows=0, allowOrphans=0)
    headings = {1: ParagraphStyle("Title", parent=body, fontName="Research-Bold", fontSize=19,
                                  leading=25, spaceAfter=15, alignment=TA_CENTER, keepWithNext=True),
                2: ParagraphStyle("Section", parent=body, fontName="Research-Bold", fontSize=12,
                                  leading=17, spaceBefore=15, spaceAfter=8, keepWithNext=True),
                3: ParagraphStyle("Subsection", parent=body, fontName="Research-Bold", fontSize=10.3,
                                  leading=15, spaceBefore=10, keepWithNext=True)}
    small = ParagraphStyle("Cell", parent=body, fontSize=7.2, leading=9.6, spaceAfter=0, alignment=0)
    reference = ParagraphStyle("Reference", parent=body, fontSize=8.8, leading=12,
                               leftIndent=12, firstLineIndent=-12, alignment=0)
    caption = ParagraphStyle("Caption", parent=body, fontSize=8.5, leading=12, alignment=TA_CENTER)
    width = A4[0] - 92
    story, lines, index = [], (PAPER / "bao_cao_nghien_cuu.md").read_text(encoding="utf-8").splitlines(), 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("|"):
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            headers, rows = block[0], block[2:]
            labels = [ALIASES.get(name, name) for name in headers]
            weights = [max(5, min(19, max([len(labels[j]), *[len(row[j]) for row in rows]]))) for j in range(len(headers))]
            fixed = {j: 46 for j, name in enumerate(headers) if name == "asset"}
            remaining = width - sum(fixed.values())
            weight_sum = sum(value for j, value in enumerate(weights) if j not in fixed)
            widths = [fixed[j] if j in fixed else remaining * value / weight_sum
                      for j, value in enumerate(weights)]
            cells = [[Paragraph("<b>"+inline(label)+"</b>", small) for label in labels]]
            cells += [[Paragraph(inline(value), small) for value in row] for row in rows]
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF0F7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#22324A")),
                ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#3566A8")),
                ("LINEBELOW", (0, -1), (-1, -1), .4, colors.HexColor("#B8C7D9")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")])]))
            story += [table, Spacer(1, 9)]
            continue
        match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if match:
            title, path = match.groups()
            source = PAPER / path
            with PILImage.open(source) as image:
                image_width = 330 if "transition" in source.name else width
                image_height = image_width * image.height / image.width
            story.append(KeepTogether([Image(str(source), image_width, image_height),
                                       Paragraph(inline(title), caption), Spacer(1, 12)]))
            index += 1
            continue
        if line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 3)
            story.append(Paragraph(inline(line.lstrip("#").strip()), headings[level]))
        else:
            paragraph = [line]
            while index+1 < len(lines) and lines[index+1].strip() and not lines[index+1].startswith(("#", "|", "![", "- ")) and not re.match(r"^\d+\.", lines[index+1]):
                index += 1
                paragraph.append(lines[index].strip())
            story.append(Paragraph(inline(" ".join(paragraph)), reference if re.match(r"^\d+\.", line) else body))
        index += 1
    output = PAPER / "bao_cao_nghien_cuu.pdf"
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Research", 7)
        canvas.setFillColor(colors.HexColor("#61718A"))
        canvas.drawString(46, 25, "AI Crypto Regime Detection | Binance | Snapshot 30/09/2026")
        canvas.drawRightString(A4[0]-46, 25, str(document.page))
        canvas.restoreState()
    document = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=46, leftMargin=46,
                                 topMargin=42, bottomMargin=42, title="Phát hiện trạng thái rủi ro của BTC và ETH",
                                 author="AI Crypto Regime Detection")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    render()
