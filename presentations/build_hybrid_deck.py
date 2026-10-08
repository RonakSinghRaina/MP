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
panel(s.shapes, 0.70, 1.75, 5.60, 3.75)
text(s.shapes, 0.95, 1.92, 5.15, 4.6, [
    [("This Week", True, RED)],
    [("•  Two U-Nets compared: ", True), ("Akeret et al.'s tf_unet (ours) vs. the U-Net of Mesarcik et al.", False)],
    [("•  Hybrid model: ", True), ("presented for the first time — a U-Net with 9 changes.", False)],
    [("•  Which changes matter: ", True), ("tested by removing parts and re-training.", False)],
], size=19, space_after=18)
panel(s.shapes, 6.65, 1.75, 5.95, 3.75)
text(s.shapes, 6.90, 1.92, 5.5, 4.6, [
    [("Key Results (LOFAR, max F1)", True, RED)],
    [("•  tf_unet: ", True), ("0.548", False)],
    [("•  Mesarcik et al. U-Net: ", True), ("0.588  (+0.039)", False)],
    [("•  Our hybrid: ", True), ("0.660 — highest of all methods", True, RED)],
    [("•  Hybrid with changes 1–3 removed: ", True), ("0.658 — no real loss", False)],
], size=19, space_after=18)

# ======================================================================= 3
s = new_slide("Two different U-Nets")
panel(s.shapes, 0.70, 1.75, 5.55, 1.55)
text(s.shapes, 0.92, 1.80, 5.15, 1.45, [
    [("Akeret et al. (2017) — tf_unet", True, NAVY)],
    "The original U-Net code for RFI detection, written in TensorFlow.",
    "This is the U-Net we ran.",
], size=15, space_after=4)
panel(s.shapes, 6.60, 1.75, 6.00, 1.55)
text(s.shapes, 6.82, 1.80, 5.6, 1.45, [
    [("Mesarcik et al. (2022) — their own U-Net", True, NAVY)],
    "Rebuilt from scratch by the authors of the LOFAR dataset paper.",
    "Same name, but a different network (see next slide).",
], size=15, space_after=4)
bar_chart(s, 2.6, 3.50, 6.9, 3.35,
          ["Akeret tf_unet\n(ours)", "Mesarcik et al.\nU-Net (published)"],
          [0.548, 0.588], [NAVY, MID], vmax=0.70, errors=[0.014, 0.003])
text(s.shapes, 9.6, 4.45, 3.0, 1.3, [
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

# ======================================================================= 5-7
# The nine changes, three per slide: a numbered badge, the name, three lines.
CHANGES = [
    ("Strip convolutions", [
        "A normal filter looks at a small 3×3 square of pixels.",
        "A strip filter looks along a long thin line instead (7, 11 or 21 pixels).",
        "Idea: RFI often appears as long streaks, so this should catch faint ones."]),
    ("Residual shortcuts", [
        "Normally the data must pass through every layer, one after another.",
        "A shortcut lets the input skip past a block and be added back at the end.",
        "Idea: faint details are not lost as the data goes deeper into the network."]),
    ("Channel attention (ECA)", [
        "The network builds many feature maps, each looking for a different pattern.",
        "Attention learns a weight for each one: useful ones up, others down.",
        "Idea: help the features that spot faint RFI stand out."]),
    ("GroupNorm", [
        "Inside a network, numbers can grow too large or shrink too small.",
        "GroupNorm rescales them after every layer so they stay in a steady range.",
        "tf_unet has no such step; this also makes training more stable."]),
    ("No ReLU on the output", [
        "tf_unet passes its final scores through a ReLU, which turns negatives into 0.",
        "With class weighting this can freeze the network so that it stops learning.",
        "The hybrid outputs its raw scores, so this trap cannot happen."]),
    ("Dice loss", [
        "The usual loss (cross-entropy) checks every pixel separately.",
        "Dice loss also checks how well the whole predicted RFI shape overlaps the true one.",
        "Helpful when RFI is rare: about 1 pixel in 130 on LOFAR."]),
    ("Same padding", [
        "tf_unet trims the border of the image at every layer (valid padding).",
        "Same padding adds a thin border so the image keeps its full size.",
        "So every pixel, including the edges, gets a prediction."]),
    ("No class weighting", [
        "Class weighting tells the model to care more about the rare RFI pixels.",
        "Here it pushed the model to over-predict RFI and hurt its final decisions.",
        "Removing it improved the score in every run (0.647 → 0.660)."]),
    ("New size and training setup", [
        "4 levels deep instead of 3, but thinner: 8 filters in the first layer, not 32.",
        "Written in PyTorch; trained with Adam, one image at a time, 28,000 steps.",
        "About 594,000 parameters in total."]),
]

for part in range(3):
    s = new_slide(f"The 9 changes in our Hybrid model ({part + 1} of 3)")
    for k in range(3):
        n = part * 3 + k
        name, lines = CHANGES[n]
        y0 = 1.78 + k * 1.62
        panel(s.shapes, 0.70, y0, 11.90, 1.48)
        badge = rect(s.shapes, 0.92, y0 + 0.44, 0.60, 0.60, RED if n < 3 else NAVY,
                     shape=MSO_SHAPE.OVAL)
        label_in(badge, str(n + 1), 18)
        text(s.shapes, 1.75, y0 + 0.06, 10.7, 1.38,
             [[(name, True, NAVY)]] + [[("•  " + ln, False)] for ln in lines],
             size=16, anchor=MSO_ANCHOR.MIDDLE, space_after=1)

# ======================================================================= 8
s = new_slide("Changes 1–3, as pictures")
COL_X = [0.70, 4.75, 8.80]
COL_W = 3.85
heads = ["1. Strip convolutions", "2. Residual shortcuts", "3. Channel attention (ECA)"]
for x0, hd in zip(COL_X, heads):
    panel(s.shapes, x0, 1.75, COL_W, 4.50)
    text(s.shapes, x0 + 0.15, 1.88, COL_W - 0.3, 0.42, [[(hd, True, NAVY)]], size=17,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)

ILL_TOP, ILL_H = 2.45, 1.65          # illustration band inside every column

# --- 1: normal 3x3 filter vs 1x7 strip, each on a 7x7 grid, centred in column 1
g = s.shapes.add_group_shape()
cell, gap = 0.19, 0.40
grid_w = 7 * cell
cx = COL_X[0] + COL_W / 2
starts = [cx - gap / 2 - grid_w, cx + gap / 2]
gy = ILL_TOP + 0.05
for gx, lab, mode in zip(starts, ["normal filter", "strip filter"], ["square", "strip"]):
    for rr in range(7):
        for cc in range(7):
            on = (mode == "square" and 2 <= rr <= 4 and 2 <= cc <= 4) or (mode == "strip" and rr == 3)
            rect(g.shapes, gx + cc * cell, gy + rr * cell, cell, cell,
                 RED if on else WHITE, line=LIGHT, lw=0.5)
    text(g.shapes, gx - 0.15, gy + grid_w + 0.08, grid_w + 0.3, 0.3, [lab], size=11, color=GREY,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)

# --- 2: input -> layers -> (+) -> out, shortcut drawn over the top, centred in column 2
g = s.shapes.add_group_shape()
cx = COL_X[1] + COL_W / 2
w_in, w_lay, d_plus, arrow = 0.75, 0.95, 0.42, 0.35
total = w_in + arrow + w_lay + arrow + d_plus + arrow
x = cx - total / 2
yb = ILL_TOP + 0.85                  # boxes' top edge
bh = 0.55
mid = yb + bh / 2
b_in = rect(g.shapes, x, yb, w_in, bh, LIGHT); label_in(b_in, "input", 11, color=INK)
x_lay = x + w_in + arrow
b_lay = rect(g.shapes, x_lay, yb, w_lay, bh, NAVY); label_in(b_lay, "layers", 11)
x_plus = x_lay + w_lay + arrow
b_plus = rect(g.shapes, x_plus, mid - d_plus / 2, d_plus, d_plus, RED, shape=MSO_SHAPE.OVAL)
label_in(b_plus, "+", 14)
line(g.shapes, x + w_in, mid, x_lay, mid, color=INK, width=1.5)
line(g.shapes, x_lay + w_lay, mid, x_plus, mid, color=INK, width=1.5)
line(g.shapes, x_plus + d_plus, mid, x_plus + d_plus + arrow, mid, color=INK, width=1.5)
top = yb - 0.45
sx1, sx2 = x + w_in / 2, x_plus + d_plus / 2
for (x1, y1, x2, y2) in [(sx1, yb, sx1, top), (sx1, top, sx2, top), (sx2, top, sx2, mid - d_plus / 2)]:
    line(g.shapes, x1, y1, x2, y2, color=RED, width=2.0)
text(g.shapes, sx1, top - 0.34, sx2 - sx1, 0.3, [[("shortcut", True, RED)]], size=11,
     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)

# --- 3: five feature maps before and after attention, centred in column 3
g = s.shapes.add_group_shape()
cx = COL_X[2] + COL_W / 2
bw, step, gap = 0.17, 0.24, 0.65
grp_w = 4 * step + bw
starts = [cx - gap / 2 - grp_w, cx + gap / 2]
base = ILL_TOP + 1.30
for k, (gx, vals, lab) in enumerate(zip(starts, [[.5] * 5, [.25, .85, .35, .95, .2]],
                                        ["before", "after"])):
    for j, v in enumerate(vals):
        hh = 1.2 * v
        rect(g.shapes, gx + j * step, base - hh, bw, hh,
             MID if k == 0 else (RED if v > 0.6 else LIGHT))
    text(g.shapes, gx - 0.1, base + 0.06, grp_w + 0.2, 0.3, [lab], size=11, color=GREY,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
rect(g.shapes, cx - 0.22, base - 0.62, 0.44, 0.26, INK, shape=MSO_SHAPE.RIGHT_ARROW)

for x0, body in zip(COL_X, [
    ["Looks along long thin lines (7, 11 or 21 pixels) instead of small squares.",
     [("Idea: ", True), ("RFI often appears as streaks.", False)]],
    ["A bypass road: the input skips past the block and is added back at the end.",
     [("Idea: ", True), ("faint details are not lost in deep layers.", False)]],
    ["Learns which feature maps matter and turns them up or down.",
     [("Idea: ", True), ("let faint-RFI features stand out.", False)]],
]):
    text(s.shapes, x0 + 0.2, 4.45, COL_W - 0.4, 1.7, body, size=15, space_after=8)

# ======================================================================= 9
s = new_slide("Do changes 1–3 matter?")
panel(s.shapes, 0.70, 1.75, 4.55, 2.30)
text(s.shapes, 0.90, 1.85, 4.2, 2.1, [
    [("Full hybrid", True, RED)],
    "All 9 changes",
    [("593,842", True), (" parameters", False)],
], size=16, space_after=6, anchor=MSO_ANCHOR.MIDDLE)
panel(s.shapes, 0.70, 4.25, 4.55, 2.40)
text(s.shapes, 0.90, 4.35, 4.2, 2.2, [
    [("Partial hybrid", True, NAVY)],
    "Changes 1–3 removed (no strips, no shortcuts, no attention): 6 changes left",
    [("486,418", True), (" parameters (18% smaller)", False)],
], size=16, space_after=6, anchor=MSO_ANCHOR.MIDDLE)
bar_chart(s, 5.75, 1.85, 6.6, 3.85, ["Full hybrid\n(9 changes)", "Partial hybrid\n(6 changes)"],
          [0.6603, 0.6585], [RED, NAVY], vmax=0.75, errors=[0.0040, 0.0047], fmt="{:.3f}")
text(s.shapes, 5.75, 5.85, 6.85, 0.8, [
    [("No — the score barely moves: 0.660 vs 0.658 ", True), ("(3 runs each, p = 0.64).", False)],
    "Strips, shortcuts and attention add 18% more parameters but no accuracy.",
], size=14, space_after=2)

# ======================================================================= 10
s = new_slide("Where everything stands on LOFAR")
bar_chart(s, 0.85, 1.80, 11.6, 4.35,
          ["σ-clip\nthreshold", "Akeret\ntf_unet", "AOFlagger", "Mesarcik\nU-Net",
           "RFI-Net", "Partial\nhybrid", "Full\nhybrid"],
          [0.410, 0.548, 0.570, 0.588, 0.598, 0.658, 0.660],
          [LIGHT, NAVY, LIGHT, MID, MID, RED, RED], vmax=0.75, label_size=12)
text(s.shapes, 0.85, 6.25, 11.6, 0.6, [
    [("max F1 on the 109 expert-labelled LOFAR images. ", False),
     ("Mesarcik et al. U-Net and RFI-Net values as published by Mesarcik et al. (2022).", False)],
], size=12, color=GREY)

# ======================================================================= 11
s = new_slide("To Do Next Week")
text(s.shapes, 1.1, 1.85, 11.0, 4.8, [
    [("•  Find what drives the gain: ", True),
     ("test changes 4–9 one at a time, starting with the Dice loss.", False)],
    [("•  Repeat runs ", True), ("where results so far come from a single run.", False)],
    [("•  Project report: ", True), ("continue from the full first draft.", False)],
], size=19, space_after=16)

prs.save(OUT)
print("saved", OUT, "|", len(prs.slides._sldIdLst), "slides")
