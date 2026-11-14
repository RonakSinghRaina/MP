"""Remove the slide's red header rule from a cropped schematic PDF.

basic_unet_schematic.pdf and hybrid_schematic.pdf are crops of slides exported from the
schematic .pptx files. Those slides carry a red header rule (an image drawn at
y = 416.52 to 421.65 pt on the 960 x 540 pt slide). The crops end at y = 416.9 pt (U-Net) and
416.5 pt (Kestrel), so the rule showed as a thin red line along the top of Fig 3.3, and, through
pixel rounding in PDF viewers, along the top of Fig 3.4 too. This deletes that one draw command
(the bytes are blanked, so the file layout stays valid) and recompresses with mutool. Nothing
else in the figure changes.

    ~/torch-env/bin/python report/Images/remove_header_band.py report/Images/hybrid_schematic.pdf
"""
import os
import shutil
import subprocess
import sys
import tempfile

BAND = b"q 954.879 0 0 5.131 5.131 416.523 cm\n/Im6 Do Q"
BLANK = b"q 954.879 0 0 5.131 5.131 416.523 cm\n        Q"


def main(pdf):
    tmp = tempfile.mkdtemp()
    raw = os.path.join(tmp, "raw.pdf")
    subprocess.run(["mutool", "clean", "-d", pdf, raw], check=True, capture_output=True)
    b = open(raw, "rb").read()
    n = b.count(BAND)
    if n == 0:
        print(pdf, ": no header band found, left unchanged")
        return
    assert n == 1, (pdf, n)
    open(raw, "wb").write(b.replace(BAND, BLANK))
    out = os.path.join(tmp, "out.pdf")
    subprocess.run(["mutool", "clean", "-gz", raw, out], check=True, capture_output=True)
    shutil.copy(out, pdf)
    shutil.rmtree(tmp)
    print(pdf, ": red header band removed")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
