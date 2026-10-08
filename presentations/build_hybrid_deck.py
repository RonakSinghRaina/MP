"""Weekly lab-meeting deck: the two U-Nets, and the hybrid model.

Built on Weekly_Report_Week2_DiL_Lab_v2.pptx so the slides inherit the lab's
branding (logos, red swoosh, footer bar, fonts, colours).

EVERY chart and diagram is drawn from plain shapes (rectangles, lines, text
boxes), not PowerPoint chart objects or images. Google Slides turns imported
PowerPoint charts into flat pictures, but plain shapes stay editable and
resizable after import. Each chart is one group: click it to move or resize it
as a unit, double-click to edit a single bar or label.

All numbers are from RFI-project-context.md (PARTS 12, 13, 18, 20, 21) and the
runs' metrics.json files.

    ~/torch-env/bin/python presentations/build_hybrid_deck.py
"""
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
TEMPLATE = os.path.join(ROOT, "Weekly_Report_Week2_DiL_Lab_v2.pptx")
OUT = os.path.join(ROOT, "Weekly_Report_Hybrid_DiL_Lab.pptx")
DATE = "08/10/2026"

RED = RGBColor(0xE3, 0x18, 0x37)
INK = RGBColor(0x1E, 0x1E, 0x1E)
NAVY = RGBColor(0x0F, 0x20, 0x43)
GREY = RGBColor(0x5A, 0x61, 0x70)
LIGHT = RGBColor(0xB9, 0xBE, 0xC9)
MID = RGBColor(0x8E, 0x9A, 0xAF)
PANEL = RGBColor(0xF6, 0xF7, 0xF9)
EDGE = RGBColor(0xD9, 0xDD, 0xE4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE_RED = RGBColor(0xFB, 0xE3, 0xE7)

prs = Presentation(TEMPLATE)
for sld in list(prs.slides._sldIdLst):        # keep masters/layouts, drop old slides
    prs.part.drop_rel(sld.rId)
    prs.slides._sldIdLst.remove(sld)
LAY_TITLE, LAY_OBJ = prs.slide_layouts[0], prs.slide_layouts[1]


# ----------------------------------------------------------------- helpers
def new_slide(title):
    s = prs.slides.add_slide(LAY_OBJ)
    tf = s.placeholders[0].text_frame
    tf.clear()
    r = tf.paragraphs[0].add_run()
    r.text = "➢ " + title
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = "Calibri", Pt(26), True, RED
    for ph in list(s.placeholders):           # drop the layout's empty footer boxes
        if ph.placeholder_format.idx != 0 and not ph.text_frame.text.strip():
            ph._element.getparent().remove(ph._element)
    return s


def text(shapes, x, y, w, h, paras, size=15, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space_after=6):
    """paras: list of paragraphs; each paragraph is a str or list of
    (text, bold[, color]) runs."""
    box = shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        runs = [(para, False)] if isinstance(para, str) else para
        for run in runs:
            t, b = run[0], run[1]
            c = run[2] if len(run) > 2 else color
            r = p.add_run()
            r.text = t
            r.font.name, r.font.size, r.font.bold, r.font.color.rgb = "Calibri", Pt(size), b, c
    return box


def rect(shapes, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, lw=1.0):
    r = shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.solid()
    r.fill.fore_color.rgb = fill
    if line is None:
        r.line.fill.background()
    else:
        r.line.color.rgb = line
        r.line.width = Pt(lw)
    r.shadow.inherit = False
    r.text_frame.text = ""
    return r


def label_in(shape, t, size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.06)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = t
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = "Calibri", Pt(size), bold, color


def line(shapes, x1, y1, x2, y2, color=GREY, width=1.25):
    c = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                             Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    return c


def panel(shapes, x, y, w, h, fill=PANEL):
    return rect(shapes, x, y, w, h, fill, line=EDGE, shape=MSO_SHAPE.ROUNDED_RECTANGLE)


def bar_chart(slide, x, y, w, h, labels, values, colors, vmax, errors=None,
              fmt="{:.3f}", label_size=12, value_size=14):
    """Vertical bar chart made of plain shapes, grouped as one object.
    Bars start at zero so their heights are honest."""
    g = slide.shapes.add_group_shape()
    sh = g.shapes
    n = len(values)
    label_h = 0.62
    plot_h = h - label_h - 0.35           # room for value labels above the bars
    base_y = y + 0.35 + plot_h
    slot = w / n
    bw = slot * 0.58
    line(sh, x, base_y, x + w, base_y, color=GREY, width=1.25)
    for i, (lab, v, col) in enumerate(zip(labels, values, colors)):
        cx = x + slot * (i + 0.5)
        bh = plot_h * v / vmax
        rect(sh, cx - bw / 2, base_y - bh, bw, bh, col)
        top = base_y - bh
        if errors and errors[i]:
            eh = plot_h * errors[i] / vmax
            line(sh, cx, top - eh, cx, top + eh, color=INK, width=1.25)
            line(sh, cx - 0.08, top - eh, cx + 0.08, top - eh, color=INK, width=1.25)
            line(sh, cx - 0.08, top + eh, cx + 0.08, top + eh, color=INK, width=1.25)
            top -= eh
        vtxt = fmt.format(v) + (f" ± {errors[i]:.3f}" if errors and errors[i] else "")
        text(sh, cx - slot / 2, top - 0.36, slot, 0.32, [[(vtxt, True, col if col in (RED,) else INK)]],
             size=value_size, align=PP_ALIGN.CENTER, space_after=0)
        text(sh, cx - slot / 2, base_y + 0.06, slot, label_h, lab.split("\n"), size=label_size,
             color=INK, align=PP_ALIGN.CENTER, space_after=0)
    return g


def table(slide, rows, x, y, w, col_w, row_h=0.42, size=13, highlight_col=None):
    nr, nc = len(rows), len(rows[0])
    t = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w),
                               Inches(row_h * nr)).table
    t.first_row = False
    t.horz_banding = False
    tot = sum(col_w)
    for i, cw in enumerate(col_w):
        t.columns[i].width = Emu(int(Inches(w) * cw / tot))
    for ri, row in enumerate(rows):
        t.rows[ri].height = Inches(row_h)
        for ci, val in enumerate(row):
            c = t.cell(ri, ci)
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if ri == 0 else (WHITE if ri % 2 else RGBColor(0xF3, 0xF5, 0xF8))
            c.margin_left = c.margin_right = Inches(0.08)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            c.text_frame.word_wrap = True
            p.alignment = PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            r.font.name, r.font.size = "Calibri", Pt(size)
            r.font.bold = ri == 0 or ci == 0
            r.font.color.rgb = WHITE if ri == 0 else (NAVY if ci == 0 else INK)
    return t


# ======================================================================= 1
s = prs.slides.add_slide(LAY_TITLE)
ph = s.placeholders[0]
ph.left, ph.top, ph.width, ph.height = Inches(1.78), Inches(4.86), Inches(10.0), Inches(0.7)
tf = ph.text_frame
tf.clear()
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = "Weekly Report – The Hybrid Model"
r.font.name, r.font.size, r.font.bold, r.font.color.rgb = "Calibri", Pt(36), True, RED
text(s.shapes, 1.78, 5.62, 10.0, 0.45, [DATE], size=21, align=PP_ALIGN.CENTER)
text(s.shapes, 0.08, 6.62, 4.4, 0.85,
     [[("Ronak Singh (BS-MS 4th Year)", True)], [("IMS23319", True)], [("School of Physics", True)]],
     size=14, space_after=0)

# ======================================================================= 2
s = new_slide("Summary")
panel(s.shapes, 0.70, 1.75, 5.60, 3.55)
text(s.shapes, 0.95, 1.92, 5.15, 4.6, [
    [("This Week", True, RED)],
    [("•  Two U-Nets compared: ", True), ("Akeret et al.'s tf_unet (ours) vs. the U-Net of Mesarcik et al.", False)],
    [("•  Hybrid model: ", True), ("presented for the first time — a U-Net with 9 changes.", False)],
    [("•  Which changes matter: ", True), ("tested by removing parts and re-training.", False)],
], size=16, space_after=12)
panel(s.shapes, 6.65, 1.75, 5.95, 3.55)
text(s.shapes, 6.90, 1.92, 5.5, 4.6, [
    [("Key Results (LOFAR, max F1)", True, RED)],
    [("•  tf_unet: ", True), ("0.548", False)],
    [("•  Mesarcik et al. U-Net: ", True), ("0.588  (+0.039)", False)],
    [("•  Our hybrid: ", True), ("0.660 — highest of all methods", True, RED)],
    [("•  Hybrid with 3 parts removed: ", True), ("0.659 — no real loss", False)],
], size=16, space_after=12)

# ======================================================================= 3
s = new_slide("Two different U-Nets")
panel(s.shapes, 0.70, 1.75, 5.55, 2.05)
text(s.shapes, 0.92, 1.85, 5.15, 1.9, [
    [("Akeret et al. (2017) — tf_unet", True, NAVY)],
    "The original U-Net code for RFI detection, written in TensorFlow.",
    "This is the U-Net we ran.",
], size=15, space_after=6)
panel(s.shapes, 6.60, 1.75, 6.00, 2.05)
text(s.shapes, 6.82, 1.85, 5.6, 1.9, [
    [("Mesarcik et al. (2022) — their own U-Net", True, NAVY)],
    "Rebuilt from scratch by the authors of the LOFAR dataset paper.",
    "Same name, but a different network (see next slide).",
], size=15, space_after=6)
bar_chart(s, 3.2, 3.95, 6.9, 2.95,
          ["Akeret tf_unet\n(ours)", "Mesarcik et al.\nU-Net (published)"],
          [0.548, 0.588], [NAVY, MID], vmax=0.70, errors=[0.014, 0.003])
text(s.shapes, 9.7, 4.6, 3.1, 1.3, [
    [("Gap: 0.039", True, RED)],
    "max F1 on the 109 expert-labelled LOFAR images",
], size=15)

# ======================================================================= 4
s = new_slide("What is different between them")
table(s, [
    ["", "Akeret et al. tf_unet (ours)", "Mesarcik et al. U-Net"],
    ["Normalisation layers", "None", "BatchNorm after every layer"],
    ["Shrinking the image", "Max-pooling (3 levels)", "Stride-2 convolutions (5 levels)"],
    ["Image edges", "Trimmed (valid padding)", "Kept (same padding)"],
    ["Network size", "465,986 parameters", "1,179,121 parameters"],
    ["Training input", "Whole 512×512 image, 4 per batch", "32×32 patches, 1024 per batch"],
    ["Input scaling", "Fixed linear range", "Clipped, then log-scaled"],
    ["Output", "2 scores, ReLU, softmax", "1 score, sigmoid"],
], x=0.85, y=1.80, w=11.6, col_w=[2.6, 4.4, 4.6], row_h=0.50, size=14)
text(s.shapes, 0.85, 6.0, 11.6, 0.8, [
    [("So the 0.039 gap compares two different networks", True, RED),
     (" — not the same U-Net trained twice.", False)],
], size=16)

# ======================================================================= 5
s = new_slide("Our Hybrid model: 9 changes to the U-Net")
rect(s.shapes, 0.70, 1.75, 5.65, 0.55, RED, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
text(s.shapes, 0.85, 1.80, 5.4, 0.45, [[("Group 1 — RFI-specific parts (3)", True, WHITE)]],
     size=17, anchor=MSO_ANCHOR.MIDDLE)
panel(s.shapes, 0.70, 2.40, 5.65, 2.35, fill=PALE_RED)
text(s.shapes, 0.90, 2.55, 5.3, 2.2, [
    [("1. Strip convolutions ", True), ("— look along long thin lines", False)],
    [("2. Residual shortcuts ", True), ("— a bypass around each block", False)],
    [("3. Channel attention (ECA) ", True), ("— turns useful features up", False)],
], size=18, space_after=16)
rect(s.shapes, 6.75, 1.75, 5.85, 0.55, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
text(s.shapes, 6.90, 1.80, 5.6, 0.45, [[("Group 2 — training & design changes (6)", True, WHITE)]],
     size=17, anchor=MSO_ANCHOR.MIDDLE)
panel(s.shapes, 6.75, 2.40, 5.85, 4.05)
text(s.shapes, 6.95, 2.52, 5.5, 4.0, [
    [("4. GroupNorm ", True), ("— keeps internal values steady", False)],
    [("5. No ReLU on the output ", True), ("— avoids the dead-network trap", False)],
    [("6. Dice loss ", True), ("— rewards the right RFI shape", False)],
    [("7. Same padding ", True), ("— predicts every pixel, edges too", False)],
    [("8. No class weighting ", True), ("— doesn't over-push towards RFI", False)],
    [("9. New size & training ", True), ("— 4 levels, thinner, PyTorch", False)],
], size=18, space_after=14)
panel(s.shapes, 0.70, 4.95, 5.65, 1.50)
text(s.shapes, 0.90, 5.05, 5.3, 1.3, [
    [("Result: ", True), ("max F1 ", False), ("0.660 ± 0.004", True, RED),
     (" on LOFAR, with 593,842 parameters.", False)],
], size=18, anchor=MSO_ANCHOR.MIDDLE)

# ======================================================================= 6
s = new_slide("The three RFI-specific parts")
cols = [0.70, 4.75, 8.80]
heads = ["Strip convolutions", "Residual shortcuts", "Channel attention (ECA)"]
for x0, hd in zip(cols, heads):
    panel(s.shapes, x0, 1.75, 3.85, 4.95)
    text(s.shapes, x0 + 0.15, 1.85, 3.55, 0.45, [[(hd, True, NAVY)]], size=17)

# --- illustration 1: normal 3x3 filter vs 1x7 strip on a 7x7 grid
g = s.shapes.add_group_shape()
cell = 0.17
for gi, (gx, label, hl) in enumerate([(0.95, "normal filter", "square"),
                                      (2.75, "strip filter", "strip")]):
    for rr in range(7):
        for cc in range(7):
            on = (hl == "square" and 2 <= rr <= 4 and 2 <= cc <= 4) or \
                 (hl == "strip" and rr == 3)
            rect(g.shapes, gx - 0.6 + cc * cell, 2.45 + rr * cell, cell, cell,
                 RED if on else WHITE, line=LIGHT, lw=0.5)
    text(g.shapes, gx - 0.75, 2.45 + 7 * cell + 0.05, 1.5, 0.3, [label], size=11,
         color=GREY, align=PP_ALIGN.CENTER, space_after=0)
text(s.shapes, 0.85, 4.25, 3.55, 2.4, [
    "Look along long thin lines (7, 11 or 21 pixels) instead of small squares.",
    [("Idea: ", True), ("RFI often appears as streaks.", False)],
], size=14)

# --- illustration 2: input -> block -> (+) -> output, with a bypass over the block
g = s.shapes.add_group_shape()
yb = 2.85
b1 = rect(g.shapes, 4.95, yb, 0.75, 0.55, LIGHT); label_in(b1, "input", 11, color=INK)
b2 = rect(g.shapes, 6.10, yb, 0.95, 0.55, NAVY); label_in(b2, "layers", 11)
b3 = rect(g.shapes, 7.45, yb + 0.07, 0.4, 0.4, RED, shape=MSO_SHAPE.OVAL); label_in(b3, "+", 14)
line(g.shapes, 5.70, yb + 0.275, 6.10, yb + 0.275, color=INK, width=1.5)
line(g.shapes, 7.05, yb + 0.275, 7.45, yb + 0.275, color=INK, width=1.5)
line(g.shapes, 7.85, yb + 0.275, 8.25, yb + 0.275, color=INK, width=1.5)
for (x1, y1, x2, y2) in [(5.33, yb, 5.33, yb - 0.4), (5.33, yb - 0.4, 7.65, yb - 0.4),
                         (7.65, yb - 0.4, 7.65, yb + 0.07)]:
    line(g.shapes, x1, y1, x2, y2, color=RED, width=2.0)
text(g.shapes, 5.8, yb - 0.75, 1.8, 0.3, [[("shortcut", True, RED)]], size=11,
     align=PP_ALIGN.CENTER, space_after=0)
text(s.shapes, 4.90, 4.25, 3.55, 2.4, [
    "A bypass road: each block passes its input straight through and only learns a small correction.",
    [("Idea: ", True), ("faint details are not lost in deep layers.", False)],
], size=14)

# --- illustration 3: channels before and after attention
g = s.shapes.add_group_shape()
before = [0.5, 0.5, 0.5, 0.5, 0.5]
after = [0.25, 0.85, 0.35, 0.95, 0.2]
for k, (vals, x0, lab) in enumerate([(before, 9.10, "before"), (after, 10.85, "after")]):
    for j, v in enumerate(vals):
        hh = 1.1 * v
        rect(g.shapes, x0 + j * 0.24, 3.65 - hh, 0.18, hh, MID if k == 0 else (RED if v > 0.6 else LIGHT))
    text(g.shapes, x0 - 0.1, 3.7, 1.35, 0.3, [lab], size=11, color=GREY,
         align=PP_ALIGN.CENTER, space_after=0)
line(g.shapes, 10.40, 3.1, 10.70, 3.1, color=INK, width=1.5)
text(s.shapes, 8.95, 4.25, 3.55, 2.4, [
    "Learns which features matter and turns them up or down.",
    [("Idea: ", True), ("let faint-RFI features stand out.", False)],
], size=14)

# ======================================================================= 7
s = new_slide("Full hybrid vs. partial hybrid")
panel(s.shapes, 0.70, 1.75, 4.55, 2.30)
text(s.shapes, 0.90, 1.85, 4.2, 2.15, [
    [("Full hybrid", True, RED)],
    "All 9 changes",
    [("593,842", True), (" parameters", False)],
], size=16, space_after=6)
panel(s.shapes, 0.70, 4.25, 4.55, 2.40)
text(s.shapes, 0.90, 4.35, 4.2, 2.25, [
    [("Partial hybrid", True, NAVY)],
    "Group 1 removed: no strips, no shortcuts, no attention (6 changes left)",
    [("486,418", True), (" parameters (18% smaller)", False)],
], size=16, space_after=6)
bar_chart(s, 5.75, 1.85, 6.6, 3.85, ["Full hybrid\n(9 changes)", "Partial hybrid\n(6 changes)"],
          [0.6603, 0.6585], [RED, NAVY], vmax=0.75, errors=[0.0040, 0.0047], fmt="{:.3f}")
text(s.shapes, 5.75, 5.85, 6.85, 0.9, [
    [("Difference: 0.002 — not significant ", True), ("(3 runs each, p = 0.64).", False)],
    "The three RFI-specific parts add 18% more parameters but no accuracy.",
], size=14, space_after=2)

# ======================================================================= 8
s = new_slide("Where everything stands on LOFAR")
bar_chart(s, 0.85, 1.80, 11.6, 4.35,
          ["σ-clip\nthreshold", "Akeret\ntf_unet", "AOFlagger", "Mesarcik\nU-Net",
           "RFI-Net", "Partial\nhybrid", "Full\nhybrid"],
          [0.410, 0.548, 0.570, 0.588, 0.598, 0.659, 0.660],
          [LIGHT, NAVY, LIGHT, MID, MID, RED, RED], vmax=0.75, label_size=12)
text(s.shapes, 0.85, 6.25, 11.6, 0.6, [
    [("max F1 on the 109 expert-labelled LOFAR images. ", False),
     ("Mesarcik et al. U-Net and RFI-Net values as published by Mesarcik et al. (2022).", False)],
], size=12, color=GREY)

# ======================================================================= 9
s = new_slide("To Do Next Week")
text(s.shapes, 1.1, 1.85, 11.0, 4.8, [
    [("•  Find what drives the gain: ", True),
     ("test the 6 Group-2 changes one at a time, starting with the Dice loss.", False)],
    [("•  Repeat runs ", True), ("where results so far come from a single run.", False)],
    [("•  Project report: ", True), ("continue from the full first draft.", False)],
], size=19, space_after=16)

prs.save(OUT)
print("saved", OUT, "|", len(prs.slides._sldIdLst), "slides")
