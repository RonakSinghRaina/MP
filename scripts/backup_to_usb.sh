#!/usr/bin/env bash
# Copy the minor project to a USB pen drive, then check that every file arrived intact.
#
#   ./scripts/backup_to_usb.sh                       # finds the plugged-in pen drive itself
#   ./scripts/backup_to_usb.sh --with-synthetic      # also the two synthetic datasets (+3 GB)
#   ./scripts/backup_to_usb.sh /run/media/ronaksingh/MYDRIVE    # or name the drive yourself
#
# Copies the whole project folder (about 4.2 GB) EXCEPT:
#   data/                                  LOFAR and HERA can be downloaded again from Zenodo
#                                          (doi:10.5281/zenodo.6724065); the synthetic data can be
#                                          regenerated (dataset_generator_v3_strength.py --seed 42),
#                                          or copied with --with-synthetic
#   analysis/hera_transfer_test/HERA_npy/  a 0.7 GB converted copy of the public HERA data
#   __pycache__/, *.pyc                    Python's own temporary files
# Included: the git history, the report, the context doc, every run result and the 2.6 GB of
# trained models (these exist ONLY on this laptop), plus Claude's memory notes for the project.
#
# Safe to run again later (e.g. after the HybridRFINet run): only new or changed files are
# copied. Nothing on the laptop is changed, and nothing on the pen drive outside the folder
# Minor_Project_backup is touched.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEMORY="$HOME/.claude/projects/-home-ronaksingh-Documents-minor-project-Minor-Project/memory"
SYN_DIRS=("$ROOT/data/synthetic/Synthetic Dataset 276x600" "$ROOT/data/synthetic/Synthetic Dataset 1024x265")

WITH_SYN=0; DEST_BASE=""
for arg in "$@"; do
    case "$arg" in
        --with-synthetic) WITH_SYN=1 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) DEST_BASE="$arg" ;;
    esac
done

# ---- find the pen drive ----------------------------------------------------------
if [ -z "$DEST_BASE" ]; then
    mounts=()
    for d in "/run/media/${USER:-$(id -un)}"/*/; do
        [ -d "$d" ] && mountpoint -q "$d" && mounts+=("${d%/}")
    done
    if [ ${#mounts[@]} -eq 0 ]; then
        echo "No pen drive found. Plug it in, wait until it appears in the file manager," >&2
        echo "then run this again." >&2
        exit 1
    elif [ ${#mounts[@]} -gt 1 ]; then
        echo "More than one USB drive is plugged in. Name the one to use, e.g.:" >&2
        for m in "${mounts[@]}"; do echo "  ./scripts/backup_to_usb.sh \"$m\"" >&2; done
        exit 1
    fi
    DEST_BASE="${mounts[0]}"
fi
DEST_BASE="${DEST_BASE%/}"
if [ ! -d "$DEST_BASE" ] || [ ! -w "$DEST_BASE" ]; then
    echo "Cannot write to $DEST_BASE" >&2; exit 1
fi
DEST="$DEST_BASE/Minor_Project_backup"
mkdir -p "$DEST/Minor Project" "$DEST/claude_memory"

RS=(rsync -rt --modify-window=2)   # -rt: pen drives cannot store Linux permissions
EXCLUDES=(--exclude=/data/ --exclude=/analysis/hera_transfer_test/HERA_npy/
          --exclude=__pycache__/ --exclude='*.pyc')
to_copy() {   # bytes rsync would still have to copy:  to_copy <rsync args...>
    "${RS[@]}" -n --stats "$@" | awk -F: '/Total transferred file size/ {gsub(/[^0-9]/, "", $2); print $2}'
}

# ---- enough space? -------------------------------------------------------------------
need=$(to_copy "${EXCLUDES[@]}" "$ROOT/" "$DEST/Minor Project/")
[ -n "$need" ] || { echo "Could not measure the backup size (rsync output changed?)." >&2; exit 1; }
if [ "$WITH_SYN" = 1 ]; then
    mkdir -p "$DEST/synthetic_data"
    syn=$(to_copy "${SYN_DIRS[@]}" "$DEST/synthetic_data/")
    [ -n "$syn" ] || { echo "Could not measure the synthetic data size." >&2; exit 1; }
    need=$(( need + syn ))
fi
free=$(df -B1 --output=avail "$DEST_BASE" | tail -1 | tr -d ' ')
margin=$(( 200 * 1024 * 1024 ))
echo "Pen drive    : $DEST_BASE"
echo "To copy      : $(numfmt --to=iec "$need")   (free on the drive: $(numfmt --to=iec "$free"))"
if [ $(( need + margin )) -gt "$free" ]; then
    echo "NOT ENOUGH SPACE on the pen drive. Free some space, or leave out --with-synthetic." >&2
    exit 1
fi

# ---- copy -----------------------------------------------------------------------------
echo; echo "Copying the project ..."
"${RS[@]}" --info=progress2 "${EXCLUDES[@]}" "$ROOT/" "$DEST/Minor Project/"
"${RS[@]}" "$MEMORY/" "$DEST/claude_memory/"
if [ "$WITH_SYN" = 1 ]; then
    echo "Copying the synthetic datasets ..."
    "${RS[@]}" --info=progress2 "${SYN_DIRS[@]}" "$DEST/synthetic_data/"
fi
cat > "$DEST/README_BACKUP.txt" <<EOF
Backup of the minor project (RFI detection, LOFAR), made $(date '+%F %H:%M').

Minor Project/   the project folder: code, scripts, report (report/), context doc
                 (RFI-project-context.md), run results and trained models (runs/), and the
                 git history (.git). The code, report and results are also on GitHub:
                 https://github.com/RonakSinghRaina/MP  -- the trained models (*.pt, *.h5,
                 ckpt/, best/) are NOT on GitHub, only here.
claude_memory/   Claude Code's memory notes for this project.
synthetic_data/  (only if copied with --with-synthetic) the 276x600 and 1024x265 datasets.

Not included, and how to get it back:
  data/lofar, data/hera   download from Zenodo: https://zenodo.org/records/6724065
                          then rebuild data/lofar/LOFAR_npy with
                          analysis/lofar_analysis/convert_pickle_to_memmap.py
  data/synthetic          regenerate with src/hybrid_rfi_package/dataset_generator_v3_strength.py
                          (--seed 42), or use synthetic_data/ above
  Python environments     ~/tf-env and ~/torch-env, rebuilt from requirements-tf.txt and
                          requirements-torch.txt (see CLAUDE.md, sections 2-4)
To restore: copy "Minor Project" back to ~/Documents/minor project/ and put data/ back.
EOF
sync

# ---- verify: compare every file's contents on both sides ----------------------------
echo; echo "Checking every copied file against the original ..."
bad=$("${RS[@]}" -n --checksum --out-format='%n' "${EXCLUDES[@]}" "$ROOT/" "$DEST/Minor Project/" | grep -v '/$' || true)
bad+=$("${RS[@]}" -n --checksum --out-format='%n' "$MEMORY/" "$DEST/claude_memory/" | grep -v '/$' || true)
if [ "$WITH_SYN" = 1 ]; then
    bad+=$("${RS[@]}" -n --checksum --out-format='%n' "${SYN_DIRS[@]}" "$DEST/synthetic_data/" | grep -v '/$' || true)
fi
if [ -n "$bad" ]; then
    echo "WARNING: these files do not match the originals -- run the script again:" >&2
    echo "$bad" | head -20 >&2
    exit 1
fi
n=$(find "$DEST" -type f | wc -l)
echo
echo "BACKUP COMPLETE and verified: $n files, $(du -sh "$DEST" | cut -f1), in $DEST"
echo "Before unplugging, click the eject button next to the pen drive in the file manager."
