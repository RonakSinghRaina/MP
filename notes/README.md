# `notes/` — personal study material, not part of the method

Everything in this folder is background reading and scratch material kept for
the author's own use. **None of it is used by any script, and none of it should
be cited as part of the project's method or results.** It was moved here from
the repository root during the publication-readiness audit so the top level
shows only the project itself.

| File | What it is |
|---|---|
| `Machine_Learning_Explained_Notes.md` / `.html` / `.pdf` | The same personal ML revision notes in three formats. Keep one format before release — three copies of one document is noise in a code repository. |
| `video_transcript.txt` | Auto-fetched transcript of a YouTube video (ID `Au1OxVSyGas`), used as background reading. |
| `get_transcript.py` | Throwaway script that fetched the above via `youtube_transcript_api`. Unrelated to RFI detection. Depends on a package that is in neither `requirements-*.txt`. |

## Before making the repository public

Consider deleting this folder entirely. A reviewer opening the repository should
find code, data-generation, and results — not revision notes and a video
transcript. If any of this material genuinely informed the method, cite the
source properly in the paper instead of shipping a scraped transcript.

Note also that the transcript is third-party content reproduced verbatim;
redistributing it has the same copyright issue as redistributing a publisher
PDF, only smaller.
