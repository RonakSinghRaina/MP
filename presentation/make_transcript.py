"""Write the presentation transcript (DOCX and PDF) from transcript.json.

    ~/torch-env/bin/python presentation/build_deck.py        # writes transcript.json
    ~/torch-env/bin/python presentation/make_transcript.py   # -> Kestrel_Presentation_Transcript.docx/.pdf

The text is the same as the speaker notes in the deck. The DOCX is made from a small HTML file
by LibreOffice, then the PDF from the DOCX.
"""
import html
import json
import os
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "transcript.json")
NAME = "Kestrel_Presentation_Transcript"
MAROON = "#920815"


def esc(s):
    return html.escape(s, quote=False)


def main():
    t = json.load(open(SRC))
    rows = t["slides"]
    spoken = [r for r in rows if r["spoken"]]
    backup = [r for r in rows if not r["spoken"]]
    first_result = next(r for r in spoken if r["title"].startswith("Experimental Results"))
    at7 = next(r for r in spoken if r["start"] and int(r["start"].split(":")[0]) * 60 +
               int(r["start"].split(":")[1]) <= 420 <= int(r["end"].split(":")[0]) * 60 + int(r["end"].split(":")[1]))
    words = sum(r["words"] for r in spoken)

    h = ["<html><head><meta charset='utf-8'><style>",
         "body{font-family:Calibri,Carlito,Arial,sans-serif;font-size:12pt;color:#262626;line-height:1.35}",
         "h1,h2,h3,span{font-family:Calibri,Carlito,Arial,sans-serif}",
         "h1{font-size:20pt;color:%s;margin:0 0 4pt 0}" % MAROON,
         "h2{font-size:14pt;color:%s;margin:16pt 0 4pt 0}" % MAROON,
         "h3{font-size:12pt;color:#262626;margin:12pt 0 2pt 0}",
         "p{margin:0 0 6pt 0} .muted{color:#595959} .time{color:#595959;font-weight:normal}",
         "table{border-collapse:collapse;margin:4pt 0 8pt 0} td,th{border:1px solid #BFBFBF;padding:3pt 6pt;"
         "font-size:11pt;text-align:left} th{background:#F2F2F2}",
         "</style></head><body>"]
    h.append("<h1><font face='Calibri'>Presentation transcript</font></h1>")
    h.append("<p class='muted'>Detecting Radio Frequency Interference in Radio Spectrograms with Convolutional "
             "Networks &middot; Ronak Singh (IMS23319) &middot; 4 December 2026</p>")

    h.append("<h2><font face='Calibri'>How to use this transcript</font></h2>")
    tips = [
        "The talk is 10 minutes, followed by a 5-minute viva. Read at a calm pace, about {} words per minute: "
        "the {} spoken words then take about {}, including a short pause at each slide change. That leaves "
        "about one minute for pointing at the slides and for small delays.".format(t["wpm"], words, t["total"]),
        "Slides 1 to 10 (introduction to experimental setup) take about {}. The results start at about {}, "
        "on slide {}.".format(first_result["start"], first_result["start"], first_result["number"]),
        "At the 7-minute warning you should be on slide {} ({}). If you are behind, keep slide {} short and "
        "move on to the conclusion.".format(at7["number"], esc(at7["title"].replace("Experimental Results: ", "")),
                                             at7["number"]),
        "Slides {} to {} are backup slides for the viva; slide {} lists them with their numbers. During the "
        "slide show, type a slide number and press Enter to jump to it.".format(
            backup[0]["number"], backup[-1]["number"], backup[0]["number"]),
        "The same text is in the speaker notes of each slide, with its start and end time.",
    ]
    h.append("<ul>" + "".join("<li>{}</li>".format(x) for x in tips) + "</ul>")

    h.append("<h2><font face='Calibri'>Time plan</font></h2>")
    h.append("<table border='1' cellspacing='0' cellpadding='4'><tr><th>Slide</th><th>Title</th><th>From</th><th>To</th><th>Words</th></tr>")
    for r in spoken:
        h.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            r["number"], esc(r["title"]), r["start"], r["end"], r["words"]))
    h.append("</table>")

    h.append("<h2><font face='Calibri'>Transcript</font></h2>")
    for r in spoken:
        h.append("<h3><font face='Calibri'>Slide {} &middot; {} <span class='time'>({} &ndash; {})</span>"
                 "</font></h3>".format(
            r["number"], esc(r["title"]), r["start"], r["end"]))
        h.append("<p>{}</p>".format(esc(r["script"])))

    h.append("<h2><font face='Calibri'>Backup slides (for the viva)</font></h2>")
    for r in backup:
        h.append("<h3><font face='Calibri'>Slide {} &middot; {}</font></h3>".format(r["number"], esc(r["title"])))
        h.append("<p class='muted'>{}</p>".format(esc(r["script"])))
    h.append("</body></html>")

    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, NAME + ".html")
    open(src, "w").write("\n".join(h))
    office = shutil.which("soffice") or shutil.which("libreoffice")
    subprocess.run([office, "--headless", "--infilter=HTML (StarWriter)", "--convert-to",
                    "docx:MS Word 2007 XML", "--outdir", tmp, src], check=True, capture_output=True)
    subprocess.run([office, "--headless", "--convert-to", "pdf", "--outdir", tmp,
                    os.path.join(tmp, NAME + ".docx")], check=True, capture_output=True)
    for ext in (".docx", ".pdf"):
        shutil.copy(os.path.join(tmp, NAME + ext), os.path.join(HERE, NAME + ext))
        print("saved", os.path.join(HERE, NAME + ext))
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
