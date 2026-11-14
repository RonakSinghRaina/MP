"""Build the final presentation on the lab template.

    ~/torch-env/bin/python presentation/make_slide_figures.py     # figures first
    ~/torch-env/bin/python presentation/build_deck.py             # -> Kestrel_RFI_Presentation.pptx

The template (template/Project_Presentation_Template.pptx) supplies everything about the look:
the title-slide design, the IISER and DIL Lab logos, the red rules, the fonts (Calibri Light /
Calibri), the red bold title style and the slide-number field. This script only adds content.

Numbers: the headline means are recomputed from the saved runs and checked (check_numbers);
the timings come from analysis/timing/*.json; the rest are quoted from the report (the table
is named next to each). The speaker notes hold the transcript, with times worked out from the
word count at TALK_WPM words per minute; transcript.json carries the same text for
make_transcript.py.
"""
import copy
import glob
import json
import os

import numpy as np
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEMPLATE = os.path.join(HERE, "template", "Project_Presentation_Template.pptx")
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "Kestrel_RFI_Presentation.pptx")
TRANSCRIPT = os.path.join(HERE, "transcript.json")
TALK_WPM = 125            # a calm, formal pace
SECONDS_PER_CHANGE = 2    # pause when changing slides

# ---- look ------------------------------------------------------------------------------
MAROON = "920815"         # the template's title-slide band
CHAR = "262626"           # body text
GREY = "4D4D4D"           # secondary text
MUTED = "767676"          # captions
LIGHT = "F2F2F2"          # card fill
TINT = "F7E5E7"           # light maroon tint for highlights
RULE = "C8C8C8"
NAVY = "2F4B7C"           # tf_unet in charts (matches the U-Net schematic boxes)
BAR = "A6A6A6"            # other methods in charts
WHITE = "FFFFFF"
MONO = "Courier New"

X0, CW = 0.75, 11.833     # content left edge and width (inches); right edge 12.583
MSG_Y = 1.86              # message line under the template's header rule
EMU = 914400


def I(v):
    return Emu(int(round(v * EMU)))


def rgb(h):
    return RGBColor.from_string(h)


# ---- numbers -----------------------------------------------------------------------------
def mean_sd(pattern, key):
    v = [json.load(open(f))[key] for f in sorted(glob.glob(os.path.join(ROOT, pattern)))]
    assert len(v) == 3, (pattern, v)
    return float(np.mean(v)), float(np.std(v, ddof=1))


def check_numbers():
    """The headline numbers on the slides, recomputed from the saved runs."""
    def near(a, b, tol=5e-5):
        assert abs(a - b) < tol, (a, b)
    m, s = mean_sd("runs/lofar/hybrid_b08_nocw_seed[012]/eval_test/metrics.json", "max_f1")
    near(m, 0.6603); near(s, 0.0040)
    m, s = mean_sd("runs/lofar/lr1e-4_seed[012]/eval_test/metrics.json", "max_f1")
    near(m, 0.5482); near(s, 0.0139)
    m, s = mean_sd("runs/hera/hybrid_b08_nocw_seed[012]/eval_test/metrics.json", "max_f1_crop472")
    near(m, 0.9442); near(s, 0.0043)
    m, s = mean_sd("runs/hera/tfunet_lr1e-4_seed[012]/eval_test/metrics.json", "max_f1")
    near(m, 0.9550); near(s, 0.0025)
    m, _ = mean_sd("runs/lofar/abl_plain_unet_seed[012]/eval_test/metrics.json", "max_f1")
    near(m, 0.6585)
    m, s = mean_sd("runs/lofar/abl_no_groupnorm_seed[012]/eval_test/metrics.json", "max_f1")
    near(m, 0.6495); near(s, 0.0178)
    m, s = mean_sd("runs/lofar/nodice_b08_nocw_seed[012]/eval_test/metrics.json", "max_f1")
    near(m, 0.6495); near(s, 0.0049)
    near(json.load(open(os.path.join(ROOT, "runs/lofar/hybrid_mesarcik_clean_bce_seed0/eval_test/metrics.json")))
         ["max_f1"], 0.6258)
    near(json.load(open(os.path.join(ROOT, "runs/lofar/mesarcik_unet_clean_seed0/eval_test/metrics.json")))
         ["max_f1"], 0.6166)
    k = json.load(open(os.path.join(ROOT, "analysis/timing/kestrel_timing.json")))
    u = json.load(open(os.path.join(ROOT, "analysis/timing/tfunet_timing.json")))
    assert k["parameters"] == 593842 and u["parameters"] == 465986
    return "{:.1f}".format(k["ms_per_image_median"]), "{:.1f}".format(u["ms_per_image_median"])


# ---- text helpers --------------------------------------------------------------------------
def style_run(run, size=None, bold=None, italic=None, color=None, font=None, sup=False):
    f = run.font
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color is not None:
        f.color.rgb = rgb(color)
    if font is not None:
        f.name = font
    if sup:
        run._r.get_or_add_rPr().set("baseline", "30000")
    t = run._r.find(qn("a:t"))
    if t is not None and t.text and (t.text != t.text.strip()):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def add_bullet(p, color=MAROON, indent=0.24):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(indent * EMU)))
    pPr.set("indent", str(-int(indent * EMU)))
    clr = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(clr, qn("a:srgbClr")).set("val", color)
    etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
    etree.SubElement(pPr, qn("a:buChar")).set("char", "•")


def fill(tf, paras, size=16, color=CHAR, bold=False, align=PP_ALIGN.LEFT, font=None,
         space_after=0, line_spacing=None):
    """paras: list of paragraphs. A paragraph is a str, a list of runs, or a dict with "runs"
    (or "text") and any of: align, size, bold, color, font, bullet, space_after, space_before,
    line_spacing. A run is a str or (text, {size, bold, italic, color, font, sup})."""
    for k, para in enumerate(paras):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        if isinstance(para, dict):
            o = para
        elif isinstance(para, str):
            o = {"runs": [para]}
        else:
            o = {"runs": para}
        runs = o.get("runs") or [o.get("text", "")]
        p.alignment = o.get("align", align)
        if o.get("space_after", space_after):
            p.space_after = Pt(o.get("space_after", space_after))
        if o.get("space_before"):
            p.space_before = Pt(o["space_before"])
        if o.get("line_spacing", line_spacing):
            p.line_spacing = o.get("line_spacing", line_spacing)
        if o.get("bullet"):
            add_bullet(p)
        elif o.get("indent"):
            p._p.get_or_add_pPr().set("marL", str(int(0.24 * EMU)))
        for r in runs:
            text, ro = (r, {}) if isinstance(r, str) else r
            run = p.add_run()
            run.text = text
            style_run(run, size=ro.get("size", o.get("size", size)),
                      bold=ro.get("bold", o.get("bold", bold)), italic=ro.get("italic"),
                      color=ro.get("color", o.get("color", color)),
                      font=ro.get("font", o.get("font", font)), sup=ro.get("sup", False))


def tbox(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, margins=(0, 0, 0, 0), name=None, **kw):
    s = slide.shapes.add_textbox(I(x), I(y), I(w), I(h))
    tf = s.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = [I(m) for m in margins]
    tf.vertical_anchor = anchor
    fill(tf, paras, **kw)
    if name:
        s.name = name
    return s


def shape(slide, kind, x, y, w, h, fill_color=None, line=None, line_w=1.0, radius=0.12, paras=None,
          anchor=MSO_ANCHOR.MIDDLE, margins=(0.12, 0.06, 0.12, 0.06), name=None, **kw):
    s = slide.shapes.add_shape(kind, I(x), I(y), I(w), I(h))
    s.shadow.inherit = False
    if fill_color:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill_color)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(line_w)
    else:
        s.line.fill.background()
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(0.5, radius / min(w, h))
    tf = s.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = [I(m) for m in margins]
    tf.vertical_anchor = anchor
    if paras:
        fill(tf, paras, **kw)
    if name:
        s.name = name
    return s


def card(slide, x, y, w, h, fill_color=LIGHT, **kw):
    return shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill_color=fill_color, **kw)


def badge(slide, cx, cy, d, text, fill_color=MAROON, size=16):
    return shape(slide, MSO_SHAPE.OVAL, cx - d / 2, cy - d / 2, d, d, fill_color=fill_color,
                 margins=(0, 0, 0, 0), paras=[{"runs": [(text, {"bold": True, "color": WHITE})],
                                               "align": PP_ALIGN.CENTER}], size=size)


def line(slide, x1, y1, x2, y2, color=RULE, width=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, I(x1), I(y1), I(x2), I(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    return c


def picture(slide, name, x, y, w=None, h=None, alt=""):
    p = slide.shapes.add_picture(os.path.join(FIG, name), I(x), I(y),
                                 I(w) if w else None, I(h) if h else None)
    p._element.nvPicPr.cNvPr.set("descr", alt)
    return p


def message(slide, text):
    """The slide's one-line message, under the template's header rule."""
    return tbox(slide, X0, MSG_Y, CW, 0.48, [text], size=22, bold=True, color=CHAR,
                anchor=MSO_ANCHOR.MIDDLE, name="Message")


def caption(slide, x, y, w, h, text, align=PP_ALIGN.LEFT):
    return tbox(slide, x, y, w, h, [text], size=12, color=MUTED, align=align, name="Caption")


def table(slide, x, y, col_w, rows, row_h=0.42, size=15, highlight=(), groups=(), aligns=None,
          header_fill=MAROON, name=None):
    """rows[0] is the header. groups: row indices that are group labels (merged, italic).
    highlight: row indices drawn in the maroon tint. A cell is a str or a list of runs."""
    n, m = len(rows), len(col_w)
    gf = slide.shapes.add_table(n, m, I(x), I(y), I(sum(col_w)), I(row_h * n))
    if name:
        gf.name = name
    tbl = gf.table
    tbl.first_row = True
    tbl.horz_banding = False
    for j, w in enumerate(col_w):
        tbl.columns[j].width = I(w)
    band = 0
    for i, row in enumerate(rows):
        tbl.rows[i].height = I(row_h)
        if i in groups:
            tbl.cell(i, 0).merge(tbl.cell(i, m - 1))
            band = 0
        else:
            band += 1
        for j in range(m):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = I(0.1)
            c.margin_top = c.margin_bottom = I(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            if i == 0:
                fc, col, b, it = header_fill, WHITE, True, False
            elif i in groups:
                fc, col, b, it = "E6E6E6", GREY, True, True
            elif i in highlight:
                fc, col, b, it = TINT, MAROON, True, False
            else:
                fc, col, b, it = (WHITE if band % 2 else "F5F5F5"), CHAR, False, False
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(fc)
            if i in groups and j > 0:
                continue
            val = row[j] if j < len(row) else ""
            runs = val if isinstance(val, list) else [val]
            al = PP_ALIGN.LEFT if (j == 0 or i in groups) else (aligns[j] if aligns else PP_ALIGN.CENTER)
            tf = c.text_frame
            tf.word_wrap = True
            fill(tf, [{"runs": runs, "align": al}], size=size, color=col, bold=b)
            if it:
                for r in tf.paragraphs[0].runs:
                    r.font.italic = True
    return tbl


# ---- slide plumbing -------------------------------------------------------------------------
class Deck:
    def __init__(self):
        self.prs = Presentation(TEMPLATE)
        self.tpl = list(self.prs.slides)
        self.lay_content = next(l for l in self.prs.slide_layouts if l.name == "Title and Content")
        self.lay_title = next(l for l in self.prs.slide_layouts if l.name == "Title Slide")
        num = [s for s in self.tpl[1].shapes
               if s.is_placeholder and s.placeholder_format.type == PP_PLACEHOLDER.SLIDE_NUMBER]
        self.sldnum = copy.deepcopy(num[0]._element)
        s1 = self.tpl[0]
        self.s1_parts = {n: copy.deepcopy(next(sh for sh in s1.shapes if sh.name == n)._element)
                         for n in ("Title 1", "TextBox 4", "TextBox 3")}
        self.order = []
        self.notes = []

    def content(self, title, reuse=None):
        """A content slide in the template's design: reuse a template slide, or a new one
        from the same layout with the same slide-number field."""
        if reuse is not None:
            s = self.tpl[reuse]
        else:
            s = self.prs.slides.add_slide(self.lay_content)
        for ph in list(s.placeholders):
            if ph.placeholder_format.idx == 1:                 # the empty body placeholder
                ph._element.getparent().remove(ph._element)
        if reuse is None:
            s.shapes._spTree.append(copy.deepcopy(self.sldnum))
        t = s.shapes.title
        t.left, t.top, t.width, t.height = I(2.40), I(0.52), I(8.95), I(1.0)
        tf = t.text_frame
        tf.clear()
        tf.paragraphs[0].add_run().text = title
        self.order.append(s)
        return s

    def note(self, slide, script, title, spoken=True):
        self.notes.append(dict(slide=slide, title=title, script=script, spoken=spoken))

    def save(self):
        lst = self.prs.slides._sldIdLst
        by_part = {self.prs.part.related_part(e.rId): e for e in lst}
        assert len(by_part) == len(self.order), (len(by_part), len(self.order))
        for e in list(lst):
            lst.remove(e)
        for s in self.order:
            lst.append(by_part[s.part])
        # the slide-number field is recomputed by PowerPoint; keep its stored text right as well
        for k, s in enumerate(self.order):
            for fld in s.shapes._spTree.iter(qn("a:fld")):
                if fld.get("type") == "slidenum" and fld.find(qn("a:t")) is not None:
                    fld.find(qn("a:t")).text = str(k + 1)
        # timings from the word counts, then the speaker notes
        t = 0.0
        rows = []
        for k, s in enumerate(self.order):
            n = next(x for x in self.notes if x["slide"] is s)
            words = len(n["script"].split()) if n["spoken"] else 0
            dur = words * 60.0 / TALK_WPM + (SECONDS_PER_CHANGE if n["spoken"] else 0)
            start, end = t, t + dur
            if n["spoken"]:
                t = end
            head = ("[{} – {}]  ".format(clock(start), clock(end)) if n["spoken"]
                    else "[Backup slide, not presented]  ")
            s.notes_slide.notes_text_frame.text = head + n["script"]
            rows.append(dict(number=k + 1, title=n["title"], script=n["script"], spoken=n["spoken"],
                             words=words, start=clock(start) if n["spoken"] else "",
                             end=clock(end) if n["spoken"] else "", seconds=round(dur)))
        self.prs.save(OUT)
        json.dump(dict(wpm=TALK_WPM, total=clock(t), slides=rows), open(TRANSCRIPT, "w"), indent=1)
        return t, rows


def clock(sec):
    sec = int(round(sec))
    return "{}:{:02d}".format(sec // 60, sec % 60)


# =============================================================================================
def build():
    ms_k, ms_u = check_numbers()          # "10.8", "17.8"
    D = Deck()

    # ---- 1. Title (template slide 1) --------------------------------------------------------
    s = D.tpl[0]
    t = next(sh for sh in s.shapes if sh.name == "Title 1")
    t.left, t.top, t.width, t.height = I(0.67), I(4.70), I(12.0), I(1.45)
    tf = t.text_frame
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p0 = tf.paragraphs[0]._p
    p1 = copy.deepcopy(p0)
    p0.addnext(p1)
    for p, text in ((p0, "Detecting Radio Frequency Interference in"),
                    (p1, "Radio Spectrograms with Convolutional Networks")):
        r = p.find(qn("a:r"))
        r.find(qn("a:t")).text = text
        r.find(qn("a:rPr")).set("sz", "3000")
        e = p.find(qn("a:endParaRPr"))
        if e is not None:
            e.set("sz", "3000")
    d = next(sh for sh in s.shapes if sh.name == "TextBox 4")
    d.left, d.width = I(4.667), I(4.0)
    d.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    d.text_frame.paragraphs[0].runs[0].text = "4 December 2026"
    nb = next(sh for sh in s.shapes if sh.name == "TextBox 3")
    nb.left, nb.width = I(0.12), I(4.45)
    for p, text in zip(nb.text_frame.paragraphs,
                       ("Ronak Singh (IMS23319)", "Supervisor: Dr. Alwin Poulose",
                        "School of Data Science, IISER TVM")):
        p.runs[0].text = text
    nb.text_frame.paragraphs[0].runs[0].font.bold = True
    D.order.append(s)
    D.note(s, "Good morning, everyone. I am Ronak Singh. I will present my minor project, done under "
              "the supervision of Dr. Alwin Poulose, on detecting radio frequency interference in radio "
              "spectrograms with convolutional networks.", "Title")

    # ---- 2. Introduction --------------------------------------------------------------------
    s = D.content("Introduction", reuse=1)
    rows = [("Signal", [("Radio telescopes record very faint signals from space.", {})]),
            ("Problem", [("Phones, satellites and aircraft use the same frequencies. This ", {}),
                         ("radio frequency interference (RFI)", {"bold": True}),
                         (" is often far stronger than the signal.", {})]),
            ("Task", [("Flag every RFI pixel automatically: an ", {}),
                      ("image segmentation", {"bold": True}), (" problem.", {})])]
    y = 2.35
    for lab, runs, n_lines in ((rows[0][0], rows[0][1], 2), (rows[1][0], rows[1][1], 4),
                               (rows[2][0], rows[2][1], 2)):
        shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, X0, y - 0.03, 1.35, 0.46, fill_color=MAROON, radius=0.08,
              margins=(0, 0, 0, 0), paras=[{"runs": [(lab, {"bold": True, "color": WHITE})],
                                            "align": PP_ALIGN.CENTER}], size=15)
        tbox(s, 2.3, y, 4.4, 0.27 * n_lines, [runs], size=17, color=CHAR, line_spacing=1.05)
        y += 0.27 * n_lines + 0.55
    picture(s, "intro_example.png", 6.98, 2.35, w=5.6,
            alt="A LOFAR spectrogram with bright RFI lines and patches, and the RFI mask a human expert drew")
    caption(s, 6.98, 5.32, 5.6, 0.55, "A real LOFAR spectrogram (test image 1). Bright lines and patches "
            "are RFI; the right panel shows the RFI marked by a human expert.")
    D.note(s, "Radio telescopes record extremely faint signals from space. But mobile phones, satellites and "
              "aircraft use the same frequencies, and this radio frequency interference, or RFI, is often "
              "far stronger than the signal. On the right is a real spectrogram from the LOFAR telescope. "
              "The bright lines and patches are RFI, and the second panel shows where a human expert marked "
              "them. Our task is to flag every RFI pixel automatically. This is an image segmentation "
              "problem.", "Introduction")

    # ---- 3. Motivation ----------------------------------------------------------------------
    s = D.content("Motivation of the Study", reuse=2)
    items = [("Simple rules miss faint RFI",
              "Classical flaggers use fixed thresholds and assume the noise is Gaussian."),
             ("Networks are tested on simulated data",
              "Real RFI is rarer and fainter, so results may not carry over."),
             ("Real labels are imperfect",
              "LOFAR's training labels come from AOFlagger, which agrees with a human expert at "
              "F1 = 0.57 only.")]
    for k, (head, body) in enumerate(items):
        x = X0 + k * (3.75 + 0.2915)
        card(s, x, 2.05, 3.75, 3.3)
        badge(s, x + 0.55, 2.6, 0.56, str(k + 1), size=18)
        tbox(s, x + 0.27, 3.08, 3.25, 0.8, [head], size=20, bold=True, color=CHAR, anchor=MSO_ANCHOR.TOP)
        tbox(s, x + 0.27, 3.92, 3.25, 1.3, [body], size=16, color=GREY)
    card(s, X0, 5.6, CW, 0.9, fill_color=TINT, margins=(0.3, 0.05, 0.3, 0.05),
         paras=[[("Our goal: ", {"bold": True, "color": MAROON}),
                 ("a network that flags RFI well in real telescope data, checked against a human expert.",
                  {"color": CHAR})]], size=20, anchor=MSO_ANCHOR.MIDDLE)
    D.note(s, "Why is this still an open problem? Classical flaggers use fixed thresholds, so they miss "
              "faint RFI. Most neural networks are developed on simulated data, where RFI is easier to see. "
              "And real training labels come from another flagger, AOFlagger, which agrees with a human "
              "expert at an F1 score of only 0.57. So we wanted a network that works on real data, checked "
              "against a human expert.", "Motivation of the Study")

    # ---- 4. Literature review (timeline) ------------------------------------------------------
    s = D.content("Literature Review", reuse=3)
    line(s, 0.95, 3.2, 12.38, 3.2, color=RULE, width=2.5)
    nodes = [("2012", "AOFlagger", "Offringa et al. [1]", "Threshold-based flagger; the standard for LOFAR", None),
             ("2015", "U-Net", "Ronneberger et al. [2]", "Network that labels every pixel of an image", None),
             ("2017", "tf_unet", "Akeret et al. [3]", "First U-Net for RFI flagging", "our baseline"),
             ("2020", "RFI-Net", "Yang et al. [4]", "Residual network for RFI at the FAST telescope", None),
             ("2022", "LOFAR benchmark", "Mesarcik et al. [5, 6]",
              "Real LOFAR data with expert test labels; several methods compared", "our benchmark"),
             ("2023", "CRF refinement", "Chen et al. [7]", "Cleans weak RFI in pulsar data", None)]
    for k, (yr, name, who, desc, chip) in enumerate(nodes):
        cx = 1.75 + k * 1.96
        tbox(s, cx - 0.925, 2.32, 1.85, 0.5, [yr], size=24, bold=True, color=MAROON, align=PP_ALIGN.CENTER)
        shape(s, MSO_SHAPE.OVAL, cx - 0.16, 3.04, 0.32, 0.32, fill_color=MAROON, line=WHITE, line_w=2.0)
        tbox(s, cx - 0.925, 3.52, 1.85, 0.45, [name], size=17, bold=True, color=CHAR, align=PP_ALIGN.CENTER)
        tbox(s, cx - 0.925, 3.95, 1.85, 0.35, [[(who, {"italic": True})]], size=13, color=MUTED,
             align=PP_ALIGN.CENTER)
        tbox(s, cx - 0.925, 4.36, 1.85, 1.3, [desc], size=14, color=GREY, align=PP_ALIGN.CENTER)
        if chip:
            shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, cx - 0.85, 5.78, 1.7, 0.42, fill_color=TINT, radius=0.08,
                  margins=(0, 0, 0, 0), paras=[{"runs": [(chip, {"bold": True, "color": MAROON})],
                                                "align": PP_ALIGN.CENTER}], size=14)
    D.note(s, "We build on the following work. AOFlagger is the standard flagger for LOFAR, and it produced "
              "our training labels. The U-Net is the standard network for labelling every pixel. Akeret and "
              "colleagues first used it for RFI; their code, tf_unet, is our baseline. And Mesarcik and "
              "colleagues released real LOFAR data with expert labels, which is our benchmark.",
           "Literature Review")

    # ---- 5. Problem statement -----------------------------------------------------------------
    s = D.content("Problem Statement", reuse=4)
    message(s, "Give every pixel of a spectrogram a probability of being RFI")
    picture(s, "problem_input.png", 1.2, 2.62, w=2.0, h=2.0, alt="Input spectrogram")
    tbox(s, 0.95, 4.7, 2.5, 0.4, ["Spectrogram, 512 × 512"], size=15, color=GREY, align=PP_ALIGN.CENTER)
    for x in (3.55, 8.75):
        shape(s, MSO_SHAPE.RIGHT_ARROW, x, 3.37, 1.05, 0.5, fill_color="BFBFBF")
    card(s, 4.9, 2.87, 3.55, 1.5, fill_color=MAROON,
         paras=[{"runs": [("Kestrel", {"bold": True, "size": 28, "color": WHITE})], "align": PP_ALIGN.CENTER},
                {"runs": [("a U-Net-based network", {"size": 16, "color": WHITE})], "align": PP_ALIGN.CENTER}])
    picture(s, "problem_output.png", 10.13, 2.62, w=2.0, h=2.0, alt="Kestrel's RFI probability map")
    tbox(s, 9.68, 4.7, 2.9, 0.4, ["RFI probability for every pixel"], size=15, color=GREY,
         align=PP_ALIGN.CENTER)
    for k, (lab, body) in enumerate((("Training: ", "6621 real LOFAR images, labelled by AOFlagger "
                                                    "(imperfect labels)"),
                                     ("Testing: ", "109 real LOFAR images, labelled by a human expert"))):
        card(s, X0 + k * (5.79 + 0.253), 5.4, 5.79, 1.15, margins=(0.3, 0.05, 0.3, 0.05),
             paras=[[(lab, {"bold": True, "color": MAROON}), (body, {"color": CHAR})]], size=17,
             anchor=MSO_ANCHOR.MIDDLE)
    D.note(s, "Our problem is therefore: given a spectrogram, give every pixel a probability of being RFI. "
              "The difficulty is that the network learns from 6621 real images whose labels come from "
              "AOFlagger and are imperfect, but it is judged on 109 images labelled by a human expert.",
           "Problem Statement")

    # ---- 6. Contributions -----------------------------------------------------------------------
    s = D.content("Contributions", reuse=5)
    items = [("0.660", "Best score on real LOFAR data. ",
              "Kestrel's max F1 is above every published method (best: 0.598)."),
             ("½", "Half the size. ",
              "With the authors' own settings, Kestrel matches their U-Net with half the parameters."),
             ("0.675", "Expert labels help. ",
              "Fine-tuning on the expert labels of about 68 images improves Kestrel further."),
             ("3 runs", "Fair evaluation. ",
              "Test-image copies removed from training; main results averaged over 3 runs.")]
    for k, (big, lead, body) in enumerate(items):
        x = X0 + (k % 2) * (5.79 + 0.253)
        y = 2.05 + (k // 2) * 2.3
        card(s, x, y, 5.79, 2.05)
        tbox(s, x + 0.3, y, 1.9, 2.05, [big], size=40, bold=True, color=MAROON, anchor=MSO_ANCHOR.MIDDLE)
        tbox(s, x + 2.3, y + 0.15, 3.25, 1.75, [[(lead, {"bold": True}), (body, {})]], size=17, color=CHAR,
             anchor=MSO_ANCHOR.MIDDLE)
    D.note(s, "We make four contributions. Our network, Kestrel, reaches an F1 of 0.660 on real LOFAR data, "
              "above every published method. With the original authors' own settings, it matches their "
              "U-Net using half the parameters. A few expert labels raise its score to 0.675. And we made the "
              "evaluation fair, by removing test-image copies from training and repeating the main experiments "
              "three times.", "Contributions")

    # ---- 7. Methodology ---------------------------------------------------------------------------
    s = D.content("Methodology", reuse=6)
    message(s, "Kestrel keeps the U shape of a U-Net and adds five changes")
    picture(s, "kestrel_schematic.png", 1.517, 2.4, w=10.3,
            alt="Kestrel architecture: four encoder levels, a bottleneck and four decoder levels with skip "
                "connections; five numbered changes")
    D.note(s, "Kestrel keeps the shape of a U-Net. The encoder, on the left, shrinks the spectrogram step by "
              "step to learn what is in it. The decoder, on the right, grows it back to full size to find "
              "where the RFI is. Skip connections carry fine detail across, and at the end every pixel gets a "
              "probability of being RFI. We added five changes, numbered in the figure: strip convolutions "
              "for line-shaped RFI, residual shortcuts, channel attention, group normalisation and raw "
              "output scores. Details are in the appendix.", "Methodology")

    # ---- 8. Datasets (tree) ------------------------------------------------------------------------
    s = D.content("Datasets", reuse=7)
    card(s, 5.167, 1.95, 3.0, 0.55, fill_color=MAROON, radius=0.08,
         paras=[{"runs": [("Three datasets", {"bold": True, "color": WHITE})], "align": PP_ALIGN.CENTER}],
         size=18)
    xs = [X0 + k * (3.75 + 0.2915) for k in range(3)]
    cxs = [x + 1.875 for x in xs]
    line(s, 6.667, 2.5, 6.667, 2.78, color="9A9A9A", width=1.75)
    line(s, cxs[0], 2.78, cxs[2], 2.78, color="9A9A9A", width=1.75)
    for cx in cxs:
        line(s, cx, 2.78, cx, 3.05, color="9A9A9A", width=1.75)
    sets = [("Synthetic", "generated by us", [("Size: ", "276 × 600 pixels"),
                                              ("Images: ", "1000 (700 / 150 / 150)"),
                                              ("RFI: ", "14.7% of pixels"), ("Labels: ", "exact")]),
            ("HERA", "simulated telescope data", [("Size: ", "512 × 512 pixels"),
                                                  ("Images: ", "351 train / 135 test (copies removed)"),
                                                  ("RFI: ", "2.75% of pixels"), ("Labels: ", "exact")]),
            ("LOFAR", "real telescope data", [("Size: ", "512 × 512 pixels"),
                                              ("Images: ", "6621 train / 735 validation / 109 test"),
                                              ("RFI: ", "0.77% of test pixels"),
                                              ("Labels: ", "AOFlagger (train), human expert (test)")])]
    for k, (name, kind, facts) in enumerate(sets):
        main = name == "LOFAR"
        card(s, xs[k], 3.05, 3.75, 3.3, fill_color=TINT if main else LIGHT, line=MAROON if main else None,
             line_w=2.0)
        tbox(s, xs[k] + 0.27, 3.25, 2.4, 0.45, [name], size=22, bold=True, color=MAROON if main else CHAR)
        tbox(s, xs[k] + 0.27, 3.7, 3.2, 0.35, [kind], size=14, color=MUTED)
        if main:
            shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, xs[k] + 3.75 - 1.6, 3.28, 1.4, 0.36, fill_color=MAROON,
                  radius=0.08, margins=(0, 0, 0, 0),
                  paras=[{"runs": [("main dataset", {"bold": True, "color": WHITE})], "align": PP_ALIGN.CENTER}],
                  size=13)
        tbox(s, xs[k] + 0.27, 4.2, 3.25, 2.35,
             [{"runs": [(a, {"bold": True}), (b, {})], "space_after": 7} for a, b in facts], size=15,
             color=CHAR)
    D.note(s, "We used three datasets. Synthetic data that we generated, and simulated data from the HERA "
              "telescope, both with exact labels. And most importantly, real LOFAR data: 6621 training images "
              "labelled by AOFlagger, and 109 test images labelled by a human expert. RFI covers less than "
              "one percent of the LOFAR test pixels, which makes it much harder.", "Datasets")

    # ---- 9. Pseudo code ------------------------------------------------------------------------------
    s = D.content("Pseudo Code", reuse=8)
    code = [("", "input : LOFAR spectrograms X, AOFlagger masks Y", ""),
            ("1", "remove the 109 test-image copies and 35 fully flagged images", ""),
            ("2", "scale every image with one fixed range (0.5th-99.5th percentile)", ""),
            ("3", "for epoch = 1 … 40:", ""),
            ("4", "    for step = 1 … 700:", "# one random image per step"),
            ("5", "        p ← softmax(Kestrel(x))", "# RFI probability per pixel"),
            ("6", "        loss ← cross-entropy(p, y) + Dice(p, y)", ""),
            ("7", "        update the weights with Adam (lr 1e-3, cosine decay)", ""),
            ("8", "    every 2 epochs: max F1 on 150 validation images → keep best", ""),
            ("9", "test the best model on the 109 expert-labelled images", "")]
    panel = card(s, 1.15, 2.0, 11.03, 4.45, fill_color="F7F7F7", line="D9D9D9", line_w=0.75, radius=0.1)
    tbox(s, 1.5, 2.2, 10.3, 0.45, ["Training and testing Kestrel"], size=18, bold=True, color=MAROON)
    paras = []
    for num, body, comment in code:
        runs = [((num + "  ") if num else "   ", {"color": "9A9A9A", "font": MONO}),
                (body, {"color": CHAR, "font": MONO})]
        if comment:                                    # comments start in the same column
            runs.append((" " * (34 - len(body)) + comment, {"color": "8C8C8C", "font": MONO}))
        paras.append({"runs": runs, "space_after": 3})
    tbox(s, 1.5, 2.78, 10.4, 3.55, paras, size=16, color=CHAR)
    D.note(s, "In short, we clean the data, scale every image with one fixed range, and train with "
              "cross-entropy plus Dice loss, keeping the best model on the validation set. We then test it "
              "on the 109 expert-labelled images.", "Pseudo Code")

    # ---- 10. Experimental setup --------------------------------------------------------------------
    s = D.content("Experimental Setup", reuse=9)
    lr = lambda e, extra="": [("Adam, lr 10", {}), (e, {"sup": True}), (extra, {})]
    table(s, X0, 2.05, [2.35, 2.75, 2.85],
          [["", "tf_unet (baseline)", "Kestrel (ours)"],
           ["Framework", "TensorFlow", "PyTorch"],
           ["Parameters", "465,986", "593,842"],
           ["Loss", "cross-entropy + L2 penalty", "cross-entropy + Dice"],
           ["Optimiser", lr("−4"), lr("−3", ", cosine decay")],
           ["Batch size", "4", "1"],
           ["Training", "150 epochs × 175 steps", "40 epochs × 700 steps"]],
          row_h=0.52, size=15, name="Settings table")
    for k, (h, b) in enumerate((("Hardware", "NVIDIA RTX 3060 Laptop GPU, 6 GB"),
                                ("Metrics", "max F1 (as in published work), AUROC, AUPRC"),
                                ("Repeats", "3 random seeds: mean ± standard deviation"))):
        card(s, 9.05, 2.05 + k * 1.27, 3.533, 1.1, margins=(0.22, 0.08, 0.22, 0.08), anchor=MSO_ANCHOR.MIDDLE,
             paras=[{"runs": [(h, {"bold": True, "color": MAROON})], "space_after": 2},
                    {"runs": [(b, {"color": CHAR})]}], size=15)
    tbox(s, X0, 5.98, CW, 0.45, [[("Test sets: ", {"bold": True, "color": MAROON}),
                                  ("109 expert-labelled LOFAR images  ·  135 simulated HERA images", {})]],
         size=17, color=CHAR, anchor=MSO_ANCHOR.MIDDLE)
    D.note(s, "Both networks were trained on one laptop GPU with the settings shown here. Our main results are "
              "the mean of three runs, and we report the maximum F1 score, as the published results do, with "
              "AUROC and AUPRC.", "Experimental Setup")

    # ---- 11. Results 1: visual ----------------------------------------------------------------------
    s = D.content("Experimental Results", reuse=10)
    message(s, "On real data, Kestrel's masks are the closest to the human expert's")
    picture(s, "visual_results.png", X0, 2.42, w=9.25,
            alt="Two LOFAR test images with the expert, AOFlagger, tf_unet and Kestrel masks and their F1")
    caption(s, X0, 6.55, 9.25, 0.28, "Centre 472 × 472 pixels; first training run; each model at its "
            "best threshold; F1 against the expert.")
    card(s, 10.3, 2.6, 2.283, 3.75, fill_color=TINT, margins=(0.18, 0.1, 0.18, 0.1), anchor=MSO_ANCHOR.MIDDLE,
         paras=[{"runs": [("93–99", {"bold": True, "size": 40, "color": MAROON})], "align": PP_ALIGN.CENTER},
                {"runs": [("of 109 test images", {"bold": True, "size": 18, "color": CHAR})],
                 "align": PP_ALIGN.CENTER, "space_after": 10},
                {"runs": [("where Kestrel's mask beats tf_unet's", {"size": 15, "color": CHAR})],
                 "align": PP_ALIGN.CENTER, "space_after": 6},
                {"runs": [("per-image F1, in each of the 3 runs", {"size": 12, "color": MUTED})],
                 "align": PP_ALIGN.CENTER}])
    D.note(s, "Now the results, starting with what the models actually flag. Here are two real test images. "
              "From left to right: the spectrogram, the expert's mask, AOFlagger's mask, and the masks from "
              "tf_unet and from Kestrel, with each mask's F1 score against the expert below it. Kestrel's "
              "masks are the closest to the expert's. On the second image, tf_unet merges the thin lines on "
              "the right into large blobs, while Kestrel keeps them thin. This holds broadly: Kestrel beats "
              "tf_unet on 93 to 99 of the 109 test images.", "Experimental Results: masks")

    # ---- 12. Results 2: benchmark chart ---------------------------------------------------------------
    s = D.content("Experimental Results")
    message(s, "Kestrel beats every published method on real LOFAR data")
    cats = ["σ-clipping (simple threshold)", "tf_unet (our run)", "AOFlagger", "U-Net, Mesarcik et al.",
            "RFI-Net", "Kestrel (ours)"]
    vals = [0.4103, 0.5482, 0.5698, 0.5876, 0.5979, 0.6603]            # report Table 5.1
    cols = ["C9C9C9", NAVY, BAR, BAR, BAR, MAROON]
    cd = CategoryChartData(number_format="0.000")
    cd.categories = cats
    cd.add_series("max F1", vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, I(X0), I(2.4), I(7.75), I(3.85), cd)
    gf.name = "LOFAR benchmark chart"
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = False
    ch.font.name = "Calibri"
    ch.font.size = Pt(15)
    ch.font.color.rgb = rgb(CHAR)
    plot = ch.plots[0]
    plot.gap_width = 45
    plot.vary_by_categories = False
    ser = plot.series[0]
    ser.invert_if_negative = False
    for k, c in enumerate(cols):
        pt = ser.points[k]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(c)
        pt.format.line.fill.background()
    dl = ser.data_labels                       # series level, so every bar gets its value
    dl.show_value = True
    dl.number_format = "0.000"
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(16)
    dl.font.color.rgb = rgb(CHAR)
    kd = ser.points[5].data_label              # Kestrel's value in bold maroon
    kd.position = XL_LABEL_POSITION.OUTSIDE_END
    kd.font.size = Pt(18)
    kd.font.bold = True
    kd.font.color.rgb = rgb(MAROON)
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0.0, 0.75
    va.has_major_gridlines = False
    va.visible = False
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(15)
    ca.tick_labels.font.color.rgb = rgb(CHAR)
    ca.major_tick_mark = XL_TICK_MARK.NONE
    ca.format.line.color.rgb = rgb("BFBFBF")
    caption(s, X0, 6.3, 7.75, 0.45, "Max F1 on the 109 expert-labelled test images. Ours: mean of 3 runs. "
            "AOFlagger, U-Net and RFI-Net: as published by Mesarcik et al. [5]; σ-clipping: computed by us.")
    for k, (big, body) in enumerate((("+0.062", "above RFI-Net, the best published method (0.598)"),
                                     ("+0.090", "above AOFlagger (0.570), even though Kestrel learned only "
                                                "from AOFlagger's labels"))):
        card(s, 8.85, 2.45 + k * 2.0, 3.733, 1.8, fill_color=TINT if k == 0 else LIGHT,
             margins=(0.25, 0.1, 0.25, 0.1), anchor=MSO_ANCHOR.MIDDLE,
             paras=[{"runs": [(big, {"bold": True, "size": 34, "color": MAROON})], "space_after": 2},
                    {"runs": [(body, {"size": 16, "color": CHAR})]}])
    D.note(s, "This chart compares all methods on the 109 expert-labelled images, using the maximum F1 score. "
              "The blue bar is our baseline, tf_unet, and the grey bars are other methods. Kestrel reaches "
              "0.660, the highest of all: 0.062 above RFI-Net, the best published method, and 0.11 above "
              "tf_unet. One point deserves attention. AOFlagger scores 0.570, and Kestrel learned only from "
              "AOFlagger's labels. Yet it ends up closer to the human expert than AOFlagger itself.",
           "Experimental Results: benchmark")

    # ---- 13. Results 3: fair comparison and speed -------------------------------------------------------
    s = D.content("Experimental Results")
    message(s, "Kestrel matches the authors' U-Net at half the size, and it is fast")
    groups = [("Same training settings as Mesarcik et al.", 0), ("Speed on our laptop GPU", 2)]
    colw, gap = 2.8, (CW - 4 * 2.8) / 3
    for text, k in groups:
        x = X0 + k * (colw + gap)
        tbox(s, x, 2.45, 2 * colw + gap, 0.4, [text], size=16, bold=True, color=GREY, align=PP_ALIGN.CENTER)
    nb = " "                               # no line break inside "512 x 512" or "10.8 ms"
    stats = [("Max F1", "0.626", "vs 0.617", "Kestrel vs their U-Net, one run each"),
             ("Parameters", "0.59" + nb + "M", "vs 1.18" + nb + "M", "Kestrel vs their U-Net"),
             ("Time per spectrogram", ms_k + nb + "ms", "vs " + ms_u + nb + "ms",
              "Kestrel vs tf_unet, one 512" + nb + "×" + nb + "512 image"),
             ("One training run", "25" + nb + "min", "vs 2" + nb + "h 40" + nb + "min", "Kestrel vs tf_unet on LOFAR")]
    for k, (lab, big, vs, desc) in enumerate(stats):
        x = X0 + k * (colw + gap)
        card(s, x, 2.95, colw, 3.0, fill_color=TINT if k < 2 else LIGHT, margins=(0.2, 0.42, 0.2, 0.15),
             anchor=MSO_ANCHOR.TOP,
             paras=[{"runs": [(lab, {"bold": True, "size": 16, "color": GREY})], "align": PP_ALIGN.CENTER,
                     "space_after": 8},
                    {"runs": [(big, {"bold": True, "size": 36, "color": MAROON})], "align": PP_ALIGN.CENTER},
                    {"runs": [(vs, {"size": 20, "color": GREY})], "align": PP_ALIGN.CENTER, "space_after": 10},
                    {"runs": [(desc, {"size": 14, "color": CHAR})], "align": PP_ALIGN.CENTER}])
    caption(s, X0, 6.12, CW, 0.4, "Time per spectrogram: median over the 109 test images, one at a time "
            "(PyTorch vs TensorFlow), RTX 3060 Laptop GPU. Training: one full run, including validation.")
    D.note(s, "A fair question is whether Kestrel wins only because of its training settings. So we also "
              "trained it with exactly the settings of Mesarcik and colleagues. It then matches their U-Net, "
              "scoring 0.626 against 0.617, with half the parameters. Kestrel is also efficient. On our laptop "
              "GPU it processes one spectrogram in " + ms_k + " milliseconds, against " + ms_u + " for tf_unet, "
              "and a complete training run takes about 25 minutes instead of 2 hours 40 minutes.",
           "Experimental Results: fair comparison and speed")

    # ---- 14. Results 4: expert labels and HERA ------------------------------------------------------------
    s = D.content("Experimental Results")
    message(s, "Expert labels help further, and Kestrel also wins on a second dataset")
    card(s, X0, 2.5, 5.75, 3.8)
    tbox(s, X0 + 0.3, 2.68, 5.2, 0.45, ["Fine-tuning with expert labels (LOFAR)"], size=18, bold=True,
         color=CHAR)
    tbox(s, X0 + 0.2, 3.25, 2.3, 0.75, ["0.660"], size=40, bold=True, color="8C8C8C", align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    shape(s, MSO_SHAPE.RIGHT_ARROW, X0 + 2.55, 3.42, 0.65, 0.42, fill_color="BFBFBF")
    tbox(s, X0 + 3.25, 3.25, 2.3, 0.75, ["0.675"], size=40, bold=True, color=MAROON, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    tbox(s, X0 + 0.2, 4.02, 2.3, 0.6, ["AOFlagger labels only"], size=14, color=GREY, align=PP_ALIGN.CENTER)
    tbox(s, X0 + 3.25, 4.02, 2.3, 0.6, ["plus expert labels of about 68 images"], size=14, color=GREY,
         align=PP_ALIGN.CENTER)
    tbox(s, X0 + 0.3, 4.85, 5.2, 1.6,
         [{"runs": ["Better in all 3 runs (+0.012 to +0.020)"], "bullet": True, "space_after": 6},
          {"runs": ["5-fold cross-validation: no image is tested by a model that saw its expert label"],
           "bullet": True}], size=15, color=CHAR)
    xr = X0 + 5.75 + 0.333
    card(s, xr, 2.5, 5.75, 3.8)
    tbox(s, xr + 0.3, 2.68, 5.2, 0.45, ["Second dataset: simulated HERA data"], size=18, bold=True, color=CHAR)
    table(s, xr + 0.25, 3.22, [3.75, 1.5],                                    # report Table 5.3
          [["Method", "max F1"],
           ["AOFlagger (published)", "0.873"],
           ["RFI-Net (published)", "0.895"],
           ["U-Net, Mesarcik et al. (published)", "0.902"],
           ["U-Net, Mesarcik et al. (our run)", "0.916"],
           ["Kestrel (ours)", "0.945"]],
          row_h=0.4, size=15, highlight=(5,), name="HERA table")
    caption(s, xr + 0.25, 5.72, 5.25, 0.8, "135 test images. Ours: Mesarcik et al.'s training settings, mean of "
            "3 runs. With our own settings tf_unet is slightly ahead (0.955 vs 0.944).")
    D.note(s, "Two further results. On the left: Kestrel learned only from AOFlagger's imperfect labels, so we "
              "fine-tuned it on the expert labels of about 68 images. Its score rose from 0.660 to 0.675, and "
              "it improved in all three runs. We used cross-validation, so no image was tested by a model that "
              "had seen its expert label. On the right is a second dataset, simulated HERA data. With the "
              "authors' settings, Kestrel again beats every published result, scoring 0.945. With our own "
              "settings, tf_unet is slightly ahead here, because HERA's labels add a thin rim around the RFI, "
              "which Kestrel learns very closely.", "Experimental Results: expert labels and HERA")

    # ---- 15. Conclusion ---------------------------------------------------------------------------------
    s = D.content("Conclusion", reuse=11)
    items = [("Best on real data: ", "Kestrel beats every published method on LOFAR (max F1 0.660 vs 0.598)."),
             ("Efficient: ", "with the authors' settings it matches their U-Net at half the size, and it flags "
                             "a spectrogram in " + ms_k + " ms."),
             ("Expert labels help: ", "about 68 expert-labelled images raise the score to 0.675."),
             ("Holds on HERA: ", "with the authors' settings Kestrel also beats the published results "
                                 "(0.945 vs 0.902).")]
    for k, (lead, body) in enumerate(items):
        y = 2.0 + k * 0.83
        badge(s, X0 + 0.27, y + 0.395, 0.52, str(k + 1), size=17)
        tbox(s, X0 + 0.8, y + 0.02, CW - 0.8, 0.75, [[(lead, {"bold": True, "color": MAROON}), (body, {})]],
             size=18, color=CHAR, anchor=MSO_ANCHOR.MIDDLE)
    tbox(s, X0, 5.4, CW, 0.5, [[("Limitations: ", {"bold": True}),
                                ("with our settings tf_unet is slightly ahead on HERA (0.955 vs 0.944); Kestrel's "
                                 "three added modules contribute little (appendix).", {})]],
         size=15, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
    card(s, X0, 6.02, CW, 0.65, fill_color=TINT, margins=(0.3, 0.04, 0.3, 0.04),
         paras=[[("Lesson: ", {"bold": True, "color": MAROON}),
                 ("test RFI methods on real data, with several runs.", {"color": CHAR})]], size=18,
         anchor=MSO_ANCHOR.MIDDLE)
    D.note(s, "To conclude. Kestrel beats every published method on real LOFAR data, with a maximum F1 of "
              "0.660. With the authors' own settings it matches their U-Net at half the size, and it is fast. "
              "A few expert labels raise its score to 0.675, and on HERA it also beats the published results. "
              "Two limitations: with our settings tf_unet is slightly ahead on HERA, and Kestrel's three added "
              "modules contribute little. The broader lesson is that RFI methods must be tested on real data, "
              "with several runs.", "Conclusion")

    # ---- 16. Future work (process chevrons) ------------------------------------------------------------------
    s = D.content("Future Work", reuse=12)
    steps = [("Explain the full gain", "Test the remaining design choices one at a time: output layer, padding, "
                                       "depth, batch size and learning rate."),
             ("Label smarter", "Find which images an expert should label first, to get the most from a few "
                               "labels."),
             ("Real pulsar data", "Apply Kestrel to real pulsar observations.")]
    shades = [MAROON, "A8323D", "BD5D66"]
    for k, (head, body) in enumerate(steps):
        x = X0 + k * (3.75 + 0.2915)                      # the same grid as the cards below
        w = 3.75 + 0.2915 if k < 2 else 3.75             # the last arrow ends at the right margin
        kind = MSO_SHAPE.PENTAGON if k == 0 else MSO_SHAPE.CHEVRON
        shape(s, kind, x, 2.55, w, 1.0, fill_color=shades[k], margins=(0.1, 0, 0.1, 0),
              paras=[{"runs": [("{}  ".format(k + 1), {"bold": True, "color": WHITE}),
                               (head, {"bold": True, "color": WHITE})], "align": PP_ALIGN.CENTER}], size=18)
        card(s, x, 3.85, 3.75, 2.0, margins=(0.25, 0.2, 0.25, 0.2), anchor=MSO_ANCHOR.TOP,
             paras=[{"runs": [(body, {"color": CHAR})]}], size=17)
    D.note(s, "In future, we would like to test the remaining design choices one at a time to explain the full "
              "gain, find which images an expert should label first, and apply Kestrel to real pulsar "
              "observations.", "Future Work")

    # ---- 17. References ------------------------------------------------------------------------------------
    s = D.content("References", reuse=13)
    refs = ["A. R. Offringa, J. J. van de Gronde, J. B. T. M. Roerdink. A morphological algorithm for improving "
            "radio-frequency interference detection. Astronomy & Astrophysics 539, A95 (2012).",
            "O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional networks for biomedical image "
            "segmentation. MICCAI, LNCS 9351, 234–241 (2015).",
            "J. Akeret, C. Chang, A. Lucchi, A. Refregier. Radio frequency interference mitigation using deep "
            "convolutional neural networks. Astronomy and Computing 18, 35–39 (2017).",
            "Z. Yang, C. Yu, J. Xiao, B. Zhang. Deep residual detection of radio frequency interference for "
            "FAST. MNRAS 492, 1421–1431 (2020).",
            "M. Mesarcik, A.-J. Boonstra, E. Ranguelova, R. V. van Nieuwpoort. Learning to detect radio "
            "frequency interference in radio astronomy without seeing it. MNRAS 516, 5367–5378 (2022).",
            "M. Mesarcik. Learning to detect RFI in radio astronomy without seeing it: datasets. Zenodo, "
            "doi:10.5281/zenodo.6724065 (2022).",
            "X. Chen, J. Han, Q. Su, Z. Yang, P. Zhou. Cleaning radio frequency interference in pulsar-folded "
            "data based on the conditional random fields with an adaptive prior. Research in Astronomy and "
            "Astrophysics 23, 104004 (2023).",
            "K. He, X. Zhang, S. Ren, J. Sun. Deep residual learning for image recognition. CVPR, "
            "770–778 (2016).",
            "Y. Wu, K. He. Group normalization. ECCV (2018).",
            "Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, Q. Hu. ECA-Net: Efficient channel attention for deep "
            "convolutional neural networks. CVPR (2020).",
            "F. Milletari, N. Navab, S.-A. Ahmadi. V-Net: Fully convolutional neural networks for volumetric "
            "medical image segmentation. 3DV (2016).",
            "D. P. Kingma, J. Ba. Adam: A method for stochastic optimization. ICLR (2015)."]
    for c in range(2):
        paras = [{"runs": [("[{}]  ".format(6 * c + k + 1), {"bold": True, "color": MAROON}), (r, {})],
                  "space_after": 7} for k, r in enumerate(refs[6 * c:6 * c + 6])]
        tbox(s, X0 + c * 6.03, 2.0, 5.8, 4.8, paras, size=13, color=CHAR)
    D.note(s, "These are the main references.", "References")

    # ---- 18. Thank you (title-slide design) ------------------------------------------------------------------
    s = D.prs.slides.add_slide(D.lay_title)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    tree = s.shapes._spTree
    t_el = copy.deepcopy(D.s1_parts["Title 1"])
    d_el = copy.deepcopy(D.s1_parts["TextBox 4"])
    n_el = copy.deepcopy(D.s1_parts["TextBox 3"])
    for el in (t_el, d_el, n_el):
        tree.append(el)
    t = next(sh for sh in s.shapes if sh.name == "Title 1")
    t.text_frame.auto_size = MSO_AUTO_SIZE.NONE
    r = t.text_frame.paragraphs[0].runs[0]
    r.text = "Thank You"
    r._r.find(qn("a:rPr")).set("sz", "4800")
    d = next(sh for sh in s.shapes if sh.name == "TextBox 4")
    d.left, d.width = I(4.667), I(4.0)
    d.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    d.text_frame.paragraphs[0].runs[0].text = "Questions are welcome"
    nb = next(sh for sh in s.shapes if sh.name == "TextBox 3")
    nb.left, nb.width = I(0.12), I(4.45)
    texts = ("Ronak Singh (IMS23319)", "Code and results:", "github.com/RonakSinghRaina/MP")
    for p, text in zip(nb.text_frame.paragraphs, texts):
        p.runs[0].text = text
    nb.text_frame.paragraphs[0].runs[0].font.bold = True
    D.order.append(s)
    D.note(s, "Thank you for your attention. I am happy to take your questions.", "Thank You")

    # ---- Appendix ---------------------------------------------------------------------------------------
    appendix = [("Appendix: U-Net", "A basic U-Net and how tf_unet differs"),
                ("Appendix: Kestrel", "Kestrel's five changes and why each was added"),
                ("Appendix: Loss & Metrics", "The losses and the scores used"),
                ("Appendix: Data Cleaning", "Fixes to the public LOFAR data before training"),
                ("Appendix: Training Labels", "How good AOFlagger's labels are, on LOFAR and HERA"),
                ("Appendix: LOFAR Table", "All LOFAR results"),
                ("Appendix: Ablation", "Removing parts of Kestrel"),
                ("Appendix: HERA Table", "All HERA results"),
                ("Appendix: Synthetic Data", "Synthetic versus real data"),
                ("Appendix: Run Time", "Computation time of the two networks")]
    first = len(D.order) + 2                       # slide number of the first appendix slide
    s = D.content("Appendix")
    message(s, "Backup slides for questions")
    for k, (title, desc) in enumerate(appendix):
        col, row = k // 5, k % 5
        x, y = X0 + col * 6.03, 2.55 + row * 0.82
        badge(s, x + 0.3, y + 0.3, 0.6, "A{}".format(k + 1), size=15)
        tbox(s, x + 0.8, y - 0.04, 4.0, 0.42, [[(title.replace("Appendix: ", ""), {"bold": True})]],
             size=17, color=CHAR, anchor=MSO_ANCHOR.MIDDLE)
        tbox(s, x + 0.8, y + 0.36, 4.3, 0.4, [desc], size=13, color=MUTED)
        tbox(s, x + 4.85, y, 0.95, 0.6, ["slide {}".format(first + k)], size=14, color=MUTED,
             align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    D.note(s, "Index of the backup slides. Jump to a slide by typing its number and pressing Enter.",
           "Appendix", spoken=False)

    # A1 U-Net
    s = D.content(appendix[0][0])
    message(s, "A basic U-Net, drawn at Kestrel's sizes (channels × height × width)")
    picture(s, "unet_schematic.png", 1.17, 2.45, w=11.0, alt="A basic U-Net with four levels and skip connections")
    caption(s, 1.17, 6.47, 11.0, 0.3, "tf_unet: the same design, but 3 levels, 32 / 64 / 128 filters and no "
            "padding (512 × 512 in, 472 × 472 out); 465,986 parameters.")
    D.note(s, "If asked how a U-Net works: the encoder halves the image at each level and doubles the number of "
              "filters; the decoder doubles it back; skip connections join the encoder's features to the "
              "decoder's at the same size. A 1 by 1 convolution and a softmax give each pixel its RFI "
              "probability.", appendix[0][0], spoken=False)

    # A2 Kestrel's changes
    s = D.content(appendix[1][0])
    message(s, "Kestrel's five changes to the U-Net, and why each was added")
    table(s, X0, 2.5, [0.55, 2.75, 4.5, 4.033],
          [["", "Change", "What it does", "Why it was added"],
           ["1", "Strip convolutions", "1×K then K×1 filters, K = 7, 11 and 21",
            "Add up the signal along line-shaped RFI"],
           ["2", "Residual blocks [8]", "Add each block's input to its output (through a 1×1 convolution)",
            "Keep weak features from fading"],
           ["3", "ECA attention [10]", "Weight each channel from 0 to 1, from the image itself",
            "Keep faint-RFI channels from being drowned out"],
           ["4", "GroupNorm [9]", "Normalise the values within each image",
            "Stable training with one image per batch"],
           ["5", "Raw output scores", "Softmax on the raw scores, with no ReLU before it",
            "tf_unet's ReLU before the softmax can stop learning"],
           ["", "Same padding", "Output the same size as the input", "Every pixel gets a prediction"]],
          row_h=0.58, size=15, aligns=[None, PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT], name="Changes table")
    caption(s, X0, 6.62, CW, 0.25, "Kestrel: 4 levels, 8 filters in the first layer, 593,842 parameters; loss: "
            "cross-entropy + Dice [11]; optimiser: Adam [12].")
    D.note(s, "The strip filters average K pixels along a line, which makes a faint line stand out from the "
              "noise about square-root-of-K times more. ECA uses only 3 learned numbers.",
           appendix[1][0], spoken=False)

    # A3 Loss and metrics
    s = D.content(appendix[2][0])
    message(s, "The losses used for training and the scores used for testing")
    card(s, X0, 2.5, 5.75, 4.15, margins=(0.3, 0.2, 0.3, 0.2), anchor=MSO_ANCHOR.TOP,
         paras=[{"runs": [("Losses", {"bold": True, "color": MAROON, "size": 18})], "space_after": 8},
                {"runs": [("Cross-entropy: ", {"bold": True}),
                          ("minus the mean log-probability of the correct class", {})], "bullet": True,
                 "space_after": 6},
                {"runs": [("Dice loss: ", {"bold": True}), ("1 − (2Σpg + 1) / (Σp + Σg + 1)", {})],
                 "bullet": True, "space_after": 2},
                {"runs": [("p = predicted RFI probability, g = 1 for RFI pixels; it depends only on the RFI "
                           "pixels, so the many clean pixels do not swamp it", {"color": GREY, "size": 14})],
                 "space_after": 6, "indent": True},
                {"runs": [("Kestrel: ", {"bold": True}), ("cross-entropy + Dice", {})], "bullet": True,
                 "space_after": 6},
                {"runs": [("tf_unet: ", {"bold": True}), ("cross-entropy + 0.001 × L2 weight penalty", {})],
                 "bullet": True}], size=16)
    card(s, X0 + 6.083, 2.5, 5.75, 4.15, margins=(0.3, 0.2, 0.3, 0.2), anchor=MSO_ANCHOR.TOP,
         paras=[{"runs": [("Scores", {"bold": True, "color": MAROON, "size": 18})], "space_after": 8},
                {"runs": [("Precision ", {"bold": True}), ("= TP / (TP + FP): flagged pixels that are RFI", {})],
                 "bullet": True, "space_after": 6},
                {"runs": [("Recall ", {"bold": True}), ("= TP / (TP + FN): RFI pixels that were found", {})],
                 "bullet": True, "space_after": 6},
                {"runs": [("F1 ", {"bold": True}), ("= 2 · precision · recall / (precision + recall)", {})],
                 "bullet": True, "space_after": 6},
                {"runs": [("Max F1: ", {"bold": True}), ("best threshold on the test set, as in published work; "
                                                         "pooled F1: threshold chosen on validation", {})],
                 "bullet": True, "space_after": 6},
                {"runs": [("AUROC, AUPRC: ", {"bold": True}), ("threshold-free; AUPRC stays low unless rare RFI "
                                                               "is found with few false alarms", {})],
                 "bullet": True, "space_after": 6},
                {"runs": [("Accuracy misleads: flagging nothing is 99.23% accurate on LOFAR", {"color": GREY})],
                 "bullet": True}], size=16)
    D.note(s, "Max F1 is optimistic because it uses the test labels to pick the threshold, but it is what the "
              "published LOFAR results use. Our headline pooled F1 for Kestrel is 0.6561.",
           appendix[2][0], spoken=False)

    # A4 Data cleaning
    s = D.content(appendix[3][0])
    message(s, "Two problems in the public LOFAR data were fixed before training")
    items = [("Test copies", "All 109 test images are byte-for-byte copies of 109 training images; only the "
                             "masks differ. Removed from training."),
             ("Fully flagged images", "35 training images are flagged as 100% RFI, and several contain only "
                                      "zeros. Removed."),
             ("One fixed scale", None)]
    for k, (head, body) in enumerate(items):
        x = X0 + k * (3.75 + 0.2915)
        card(s, x, 2.5, 3.75, 2.75)
        badge(s, x + 0.55, 3.02, 0.56, str(k + 1), size=18)
        tbox(s, x + 0.27, 3.45, 3.25, 0.45, [head], size=19, bold=True, color=CHAR)
        if body:
            tbox(s, x + 0.27, 3.95, 3.25, 1.25, [body], size=15, color=GREY)
        else:
            tbox(s, x + 0.27, 3.95, 3.25, 1.25,
                 [[("Raw values run from 0 to 1.4 × 10", {}), ("11", {"sup": True}),
                   (". Every image uses one range: the 0.5th–99.5th percentiles of 200 training images.", {})]],
                 size=15, color=GREY)
    flow = [("7500", "training images"), ("7356", "after cleaning"), ("6621 + 735", "training + validation"),
            ("109", "expert-labelled test images")]
    for k, (big, small) in enumerate(flow):
        x = X0 + k * 3.02
        card(s, x, 5.55, 2.6, 1.05, fill_color=TINT if k == 3 else LIGHT, margins=(0.1, 0.05, 0.1, 0.05),
             paras=[{"runs": [(big, {"bold": True, "size": 20, "color": MAROON})], "align": PP_ALIGN.CENTER},
                    {"runs": [(small, {"size": 13, "color": GREY})], "align": PP_ALIGN.CENTER}])
        if k < 3:
            shape(s, MSO_SHAPE.RIGHT_ARROW, x + 2.66, 5.92, 0.3, 0.3, fill_color="BFBFBF")
    D.note(s, "The test-image copies matter: Mesarcik et al.'s own code trains on them. When we ran their code "
              "without them it scored 0.6166, against 0.5876 in their paper.", appendix[3][0], spoken=False)

    # A5 Training labels
    s = D.content(appendix[4][0])
    message(s, "AOFlagger's labels are imperfect, in different ways on LOFAR and HERA")
    picture(s, "label_pairs.png", X0, 2.42, h=4.3, alt="Four LOFAR test images with AOFlagger and expert masks")
    tx = X0 + 3.6
    table(s, tx, 2.5, [5.0, 1.6, 1.6],                                          # report Table 5.4
          [["", "LOFAR", "HERA"],
           ["Data", "real", "simulated"],
           ["Training images", "6621", "316"],
           ["RFI pixels in the test images", "0.77%", "2.65%"],
           ["AOFlagger vs the true labels (F1)", "0.57", "0.91"],
           ["Pixels flagged by AOFlagger / true RFI", "1.00×", "1.10×"],
           ["AOFlagger false alarms within 2 pixels of real RFI", "48%", "75%"]],
          row_h=0.48, size=15, name="Labels table")
    tbox(s, tx, 6.0, 8.2, 0.7, ["On HERA, AOFlagger flags a thin rim of extra pixels around real RFI. Kestrel "
                                "learns its labels closely and copies the rim; tf_unet copies far less of it."],
         size=14, color=GREY)
    D.note(s, "Left: LOFAR test images with AOFlagger's mask and the expert's mask; 'agreement' is the F1 between "
              "the two masks.", appendix[4][0], spoken=False)

    # A6 LOFAR table (report Table 5.1)
    s = D.content(appendix[5][0])
    table(s, X0, 1.98, [5.233, 2.2, 2.2, 2.2],
          [["Method", "max F1", "AUROC", "AUPRC"],
           ["Our training settings"],
           ["tf_unet (3 runs)", "0.5482 ± 0.0139", "0.9470 ± 0.0074", "0.5562 ± 0.0152"],
           ["Kestrel (3 runs)", "0.6603 ± 0.0040", "0.9378 ± 0.0111", "0.6583 ± 0.0088"],
           ["Mesarcik et al.'s training settings"],
           ["tf_unet, closest (3 runs)", "0.5551 ± 0.0155", "0.9482 ± 0.0052", "0.5608 ± 0.0167"],
           ["U-Net, Mesarcik et al. (1 run)", "0.6166", "0.8213", "0.5283"],
           ["Kestrel (1 run)", "0.6258", "0.9719", "0.6401"],
           ["Published by Mesarcik et al. [5]"],
           ["AOFlagger", "0.5698", "0.7883", "0.5716"],
           ["U-Net, Mesarcik et al.", "0.5876 ± 0.0031", "0.8017 ± 0.0058", "0.5920 ± 0.0031"],
           ["RFI-Net", "0.5979 ± 0.0012", "0.8109 ± 0.0037", "0.5991 ± 0.0038"],
           ["Simple threshold, no training"],
           ["σ-clipping", "0.4103", "0.7029", "0.3165"]],
          row_h=0.335, size=14, groups=(1, 4, 8, 12), highlight=(3, 7), name="LOFAR table")
    caption(s, X0, 6.72, CW, 0.25, "109 expert-labelled LOFAR test images; every learned method was trained only "
            "on AOFlagger's labels.")
    D.note(s, "Report Table 5.1. Kestrel's AUROC is slightly below tf_unet's because the Dice loss pushes outputs "
              "towards 0 or 1; without Dice its AUROC is 0.987.", appendix[5][0], spoken=False)

    # A7 Ablation
    s = D.content(appendix[6][0])
    message(s, "Removing parts of Kestrel: the three added modules contribute little")
    table(s, X0, 2.5, [4.0, 1.4, 1.3],                                      # report Table 6.1
          [["Variant (one run each)", "max F1", "Change"],
           ["Full model", "0.6624", "—"],
           ["without strip convolutions", "0.6637", "+0.0013"],
           ["without channel attention", "0.6557", "−0.0067"],
           ["without residual blocks", "0.6569", "−0.0055"],
           ["without all three (partial Kestrel)", "0.6605", "−0.0019"],
           ["partial Kestrel without GroupNorm", "0.6289", "−0.0335"]],
          row_h=0.48, size=15, name="Ablation table")
    card(s, 7.75, 2.5, 4.833, 4.15, margins=(0.3, 0.2, 0.3, 0.2), anchor=MSO_ANCHOR.TOP,
         paras=[{"runs": [("With three runs each", {"bold": True, "color": MAROON, "size": 18})],
                 "space_after": 8},
                {"runs": [("Kestrel: 0.6603 ± 0.0040", {})], "bullet": True, "space_after": 4},
                {"runs": [("Partial Kestrel: 0.6585 ± 0.0047 (the three modules add only 0.002)", {})],
                 "bullet": True, "space_after": 4},
                {"runs": [("Without GroupNorm as well: 0.6495 ± 0.0178 (a 4× larger spread)", {})],
                 "bullet": True, "space_after": 4},
                {"runs": [("Kestrel without Dice loss: 0.6495 ± 0.0049 (Dice adds 0.011)", {})],
                 "bullet": True, "space_after": 4},
                {"runs": [("A CRF on top did not help: 0.6592 → 0.4884 with Chen et al.'s settings, "
                           "+0.0012 at best", {})], "bullet": True}], size=15)
    D.note(s, "Report Table 6.1 lists one run per variant. With three runs, partial Kestrel and the version "
              "without GroupNorm differ by 0.009, which is not significant (p = 0.48), but without GroupNorm the "
              "results vary four times more between runs. The rest of the 0.11 gain over tf_unet lies in the "
              "output layer, padding, depth and width, batch size, learning rate and framework, not tested one "
              "by one.", appendix[6][0], spoken=False)

    # A8 HERA table (report Table 5.3)
    s = D.content(appendix[7][0])
    table(s, X0, 1.98, [5.233, 2.2, 2.2, 2.2],
          [["Method", "max F1", "AUROC", "AUPRC"],
           ["Our training settings"],
           ["tf_unet", "0.9550 ± 0.0025", "0.9971 ± 0.0004", "0.9698 ± 0.0027"],
           ["Kestrel", "0.9442 ± 0.0043", "0.9983 ± 0.0007", "0.9797 ± 0.0022"],
           ["Mesarcik et al.'s training settings"],
           ["tf_unet, closest", "0.9533 ± 0.0019", "0.9955 ± 0.0028", "0.9664 ± 0.0068"],
           ["U-Net, Mesarcik et al.", "0.9160 ± 0.0062", "0.9727 ± 0.0022", "0.8991 ± 0.0073"],
           ["Kestrel", "0.9450 ± 0.0020", "0.9980 ± 0.0004", "0.9757 ± 0.0056"],
           ["Published by Mesarcik et al. [5]"],
           ["AOFlagger", "0.8729", "0.9736", "0.8799"],
           ["U-Net, Mesarcik et al.", "0.9015 ± 0.0074", "0.9746 ± 0.0009", "0.8963 ± 0.0074"],
           ["RFI-Net", "0.8950 ± 0.0068", "0.9728 ± 0.0007", "0.8898 ± 0.0086"],
           ["No training"],
           ["AOFlagger, run by us", "0.8740", "0.9739", "0.8808"]],
          row_h=0.335, size=14, groups=(1, 4, 8, 12), highlight=(7,), name="HERA full table")
    caption(s, X0, 6.72, CW, 0.25, "135 HERA test images, mean ± sd over 3 runs. With our settings Kestrel "
            "is scored on the centre 472 × 472 pixels, like tf_unet.")
    D.note(s, "Report Table 5.3. With our settings tf_unet leads on max F1, Kestrel on AUROC and AUPRC.",
           appendix[7][0], spoken=False)

    # A9 Synthetic vs real
    s = D.content(appendix[8][0])
    message(s, "Results on synthetic data do not carry over to real data")
    for k, (big, small, col) in enumerate((("0.93", "tf_unet on our synthetic data", "8C8C8C"),
                                           ("0.49", "the same code and settings on real LOFAR data", MAROON))):
        x = X0 + 0.6 + k * 4.6
        card(s, x, 2.6, 3.6, 2.2, fill_color=TINT if k else LIGHT, margins=(0.2, 0.1, 0.2, 0.1),
             paras=[{"runs": [(big, {"bold": True, "size": 54, "color": col})], "align": PP_ALIGN.CENTER},
                    {"runs": [(small, {"size": 15, "color": CHAR})], "align": PP_ALIGN.CENTER}])
    shape(s, MSO_SHAPE.RIGHT_ARROW, X0 + 4.35, 3.5, 0.7, 0.45, fill_color="BFBFBF")
    tbox(s, X0 + 9.1, 2.6, 2.75, 2.2, [[("Kestrel", {"bold": True}), (" scores 0.975 on the same synthetic "
                                                                       "data.", {})]],
         size=16, color=CHAR, anchor=MSO_ANCHOR.MIDDLE)
    tbox(s, X0, 5.2, CW, 1.4,
         [{"runs": ["Synthetic labels are exact, and every labelled pixel lies above the noise by construction."],
           "bullet": True, "space_after": 6},
          {"runs": ["On LOFAR much RFI is faint, and 44% of the pixels flagged in the training labels are not RFI "
                    "according to the expert."], "bullet": True}], size=16, color=CHAR)
    D.note(s, "Max F1: tf_unet 0.9317 on the 276 by 600 synthetic data, 0.4901 on LOFAR at learning rate 1e-3 and "
              "60 epochs; Kestrel 0.9749 on synthetic data.", appendix[8][0], spoken=False)

    # A10 Run time
    s = D.content(appendix[9][0])
    message(s, "Kestrel is faster per image and much faster to train")
    table(s, X0, 2.5, [5.0, 3.4, 3.433],
          [["", "Kestrel", "tf_unet"],
           ["Parameters", "593,842", "465,986"],
           ["Time per 512 × 512 spectrogram", ms_k + " ms", ms_u + " ms"],
           ["One training run, LOFAR", "about 25 min", "about 2 h 40 min"],
           ["One training run, HERA", "about 20 min", "about 2 h 40 min"],
           ["Framework", "PyTorch 2.13", "TensorFlow 2.21"]],
          row_h=0.5, size=16, name="Run time table")
    tbox(s, X0, 5.75, CW, 0.9,
         [{"runs": ["NVIDIA RTX 3060 Laptop GPU (6 GB) on mains power. Time per spectrogram: median over the 109 "
                    "LOFAR test images, one image at a time, after 10 warm-up passes."], "bullet": True,
           "space_after": 4},
          {"runs": ["Training: wall-clock time of one complete run, including validation, with the settings "
                    "behind the reported results."], "bullet": True}], size=14, color=GREY)
    D.note(s, "Measured with analysis/timing/time_kestrel.py and time_tfunet.py. The two networks run in "
              "different frameworks, so part of the speed difference may come from the framework.",
           appendix[9][0], spoken=False)

    total, rows = D.save()
    print("saved", OUT)
    print("spoken time at {} wpm: {}".format(TALK_WPM, clock(total)))
    for r in rows:
        print("{:>2}  {:<48} {:>4} words  {:>5} – {:>5}".format(r["number"], r["title"][:48], r["words"],
                                                                    r["start"], r["end"]))


if __name__ == "__main__":
    build()
