# Modified from https://github.com/amyxlase/relpose-plus-plus/blob/main/preprocess/preprocess_co3d.py

"""
Usage:
    python -m preprocess.preprocess_co3d --category all \
        --co3d_v2_dir /path/to/co3d_v2
사용법 예시 (권장)

1. 기본 (원래와 동일)
python -m preprocess.preprocess_co3d --category car --co3d_v2_dir /path/to/co3d_v2 --output_dir /path/to/anno_dir

2. 품질 필터 무시해서 가능한 모든 시퀀스 포함
python -m preprocess.preprocess_co3d --category car --co3d_v2_dir /path/to/co3d_v2 --output_dir /path/to/anno_dir --ignore_quality

3. 특정 시퀀스 강제 포함 (쉼표 구분)
python -m preprocess.preprocess_co3d --category car --co3d_v2_dir /path/to/co3d_v2 --output_dir /path/to/anno_dir --include_sequences "seq_0001,seq_0010"

4. 포함할 시퀀스 목록을 텍스트 파일로 지정 (파일에 한 줄에 하나의 시퀀스명)
python -m preprocess.preprocess_co3d --category car --co3d_v2_dir /path/to/co3d_v2 --output_dir /path/to/anno_dir --include_sequences /path/to/include_list.txt

5. 다른 subset list 파일(예: 전체 리스트)를 쓰고 싶다면
python -m preprocess.preprocess_co3d --category car --co3d_v2_dir /path/to/co3d_v2 --output_dir /path/to/anno_dir --subset_lists_file custom_lists.json

"""
import argparse
import gzip
import json
import os
import os.path as osp
from glob import glob

import ipdb
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# fmt: off
CATEGORIES = [
    "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove",
    "bench", "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
    "cellphone", "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
    "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave", "motorcycle",
    "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
    "skateboard", "stopsign", "suitcase", "teddybear", "toaster", "toilet", "toybus",
    "toyplane", "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
]
# fmt: on


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="apple")
    parser.add_argument("--output_dir", type=str, default="data/co3d_v2_annotations")
    parser.add_argument("--co3d_v2_dir", type=str, default="data/co3d_v2")
    parser.add_argument(
        "--min_quality",
        type=float,
        default=0.5,
        help="Minimum viewpoint quality score.",
    )
    # NEW options
    parser.add_argument(
        "--subset_lists_file",
        type=str,
        default=None,
        help="Optional: override subset list JSON (default: set_lists/set_lists_fewview_dev.json inside category folder).",
    )
    parser.add_argument(
        "--ignore_quality",
        action="store_true",
        help="If set, ignore viewpoint_quality_score filter and keep all sequences from subset lists.",
    )
    parser.add_argument(
        "--include_sequences",
        type=str,
        default=None,
        help="Optional: comma-separated list of sequence names to force-include (or path to a text file with one sequence per line).",
    )
    return parser




def _load_include_sequences(arg):
    """Parse include_sequences argument: either comma list or path to txt file."""
    if arg is None:
        return set()
    if osp.isfile(arg):
        seqs = set()
        with open(arg, "r") as f:
            for line in f:
                name = line.strip()
                if name:
                    seqs.add(name)
        return seqs
    else:
        parts = [p.strip() for p in arg.split(",") if p.strip()]
        return set(parts)


def process_poses(co3d_dir, category, output_dir, min_quality, subset_lists_file=None, ignore_quality=False, include_sequences=None):
    """
    Args:
        include_sequences: set of sequence names to force-include (can be empty set)
        subset_lists_file: path to subset lists file. If None, default inside category dir is used.
        ignore_quality: if True, do not filter sequences by viewpoint_quality_score
    """
    category_dir = osp.join(co3d_dir, category)
    print("Processing category:", category)
    frame_file = osp.join(category_dir, "frame_annotations.jgz")
    sequence_file = osp.join(category_dir, "sequence_annotations.jgz")
    if subset_lists_file is None:
        subset_lists_file = osp.join(category_dir, "set_lists/set_lists_fewview_dev.json")
    else:
        # if relative path given, make it relative to category_dir if exists
        if not osp.isabs(subset_lists_file):
            candidate = osp.join(category_dir, subset_lists_file)
            if osp.exists(candidate):
                subset_lists_file = candidate

    # sanity checks
    if not osp.exists(frame_file):
        raise FileNotFoundError(f"frame file not found: {frame_file}")
    if not osp.exists(sequence_file):
        raise FileNotFoundError(f"sequence file not found: {sequence_file}")
    if not osp.exists(subset_lists_file):
        raise FileNotFoundError(f"subset lists file not found: {subset_lists_file}")

    # bbox_file = osp.join(output_dir, f"{category}_bbox.jgz")

    with open(subset_lists_file) as f:
        subset_lists_data = json.load(f)

    with gzip.open(sequence_file, "r") as fin:
        sequence_data = json.loads(fin.read())

    with gzip.open(frame_file, "r") as fin:
        frame_data = json.loads(fin.read())

    # with gzip.open(bbox_file, "r") as fin:
        # bbox_data = json.loads(fin.read())

    # build frame_data lookup by sequence -> frame_number -> frame_entry
    frame_data_processed = {}
    for f_data in frame_data:
        sequence_name = f_data["sequence_name"]
        if sequence_name not in frame_data_processed:
            frame_data_processed[sequence_name] = {}
        frame_data_processed[sequence_name][f_data["frame_number"]] = f_data

    # sequences that pass quality filter (unless ignore_quality)
    good_quality_sequences = set()
    for seq_data in sequence_data:
        if seq_data.get("viewpoint_quality_score", 0.0) > min_quality:
            good_quality_sequences.add(seq_data["sequence_name"])

    # prepare include_sequences set
    include_sequences = include_sequences or set()

    for subset in ["train", "test"]:
        category_data = {}  # {sequence_name: [{filepath, R, T, ...}]}

        # iterate subset list entries
        for seq_name, frame_number, filepath in subset_lists_data.get(subset, []):
            # if this sequence is in include_sequences, or passes quality (or ignore_quality)
            if (not ignore_quality) and (seq_name not in good_quality_sequences) and (seq_name not in include_sequences):
                continue

            if seq_name not in category_data:
                category_data[seq_name] = []

            # if frame_data_processed has this frame, use viewpoint from there (safer)
            frame_entry = None
            if seq_name in frame_data_processed and frame_number in frame_data_processed[seq_name]:
                frame_entry = frame_data_processed[seq_name][frame_number]

            if frame_entry is None:
                # fallback to what's in subset list (filepath available), but we can't get viewpoint R/T
                # skip if viewpoint not available
                print(f"[WARN] Missing frame annotation for {seq_name} frame {frame_number}, skipping this frame.")
                continue

            category_data[seq_name].append(
                {
                    "filepath": filepath,
                    "R": frame_entry["viewpoint"]["R"],
                    "T": frame_entry["viewpoint"]["T"],
                    "focal_length": frame_entry["viewpoint"].get("focal_length"),
                    "principal_point": frame_entry["viewpoint"].get("principal_point"),
                    # "bbox": bbox,
                }
            )

        # Ensure any include_sequences that were not in subset_lists_data are also added
        missing_in_subset = [s for s in include_sequences if s not in category_data]
        for s in missing_in_subset:
            if s not in frame_data_processed:
                print(f"[WARN] include_sequences requested '{s}' but no frame annotations found. Skipping.")
                continue

            # Try to create entries from frame_data_processed: prefer using 'filepath' if present,
            # otherwise try to locate image files on disk under category_dir/sequence.
            frames_dict = frame_data_processed[s]
            entries = []

            # First pass: if frame entries already contain 'filepath', use them
            for fn in sorted(frames_dict.keys()):
                fentry = frames_dict[fn]
                if "filepath" in fentry and fentry["filepath"]:
                    entries.append({
                        "filepath": fentry["filepath"],
                        "R": fentry["viewpoint"]["R"],
                        "T": fentry["viewpoint"]["T"],
                        "focal_length": fentry["viewpoint"].get("focal_length"),
                        "principal_point": fentry["viewpoint"].get("principal_point"),
                    })
                else:
                    entries.append(None)  # placeholder for second pass

            # If all entries are valid, keep them
            if all(e is not None for e in entries):
                category_data[s] = entries
                print(f"[INFO] Force-included sequence '{s}' using filepaths from frame_annotations ({len(entries)} frames).")
                continue

            # Second pass: try to locate image files under the sequence folder
            # common subfolders: 'images', 'image', 'frames'
            seq_dir_candidates = [
                osp.join(co3d_dir, category, s),
                osp.join(co3d_dir, category, s, "images"),
                osp.join(co3d_dir, category, s, "image"),
                osp.join(co3d_dir, category, s, "frames"),
            ]
            found_files = []
            for cand in seq_dir_candidates:
                if osp.exists(cand) and osp.isdir(cand):
                    found_files = sorted(glob(osp.join(cand, "*.*")))
                    # filter image extensions
                    found_files = [p for p in found_files if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
                    if found_files:
                        break

            # fallback: glob recursively under sequence directory
            if not found_files:
                base_seq_dir = osp.join(co3d_dir, category, s)
                if osp.exists(base_seq_dir):
                    found_files = sorted(glob(osp.join(base_seq_dir, "**", "*.*"), recursive=True))
                    found_files = [p for p in found_files if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]

            # If we have files, attempt to map frame_number -> file
            if found_files:
                # Option A: try to match filename containing frame number
                mapped = {}
                for fn in sorted(frames_dict.keys()):
                    fentry = frames_dict[fn]
                    # try to find a filename that contains the frame number as substring
                    candidate = None
                    for p in found_files:
                        if f"/{fn}" in p or f"_{fn}" in p or f"-{fn}" in p:
                            candidate = p
                            break
                    if candidate is None:
                        # fallback: use index-based mapping if sizes align
                        if len(found_files) > fn:
                            candidate = found_files[fn]
                    if candidate is not None:
                        mapped[fn] = candidate

                # build entries using mapped files where available
                entries2 = []
                for fn in sorted(frames_dict.keys()):
                    fentry = frames_dict[fn]
                    if fn in mapped:
                        entries2.append({
                            "filepath": mapped[fn],
                            "R": fentry["viewpoint"]["R"],
                            "T": fentry["viewpoint"]["T"],
                            "focal_length": fentry["viewpoint"].get("focal_length"),
                            "principal_point": fentry["viewpoint"].get("principal_point"),
                        })
                    else:
                        # skip frames with no mapping
                        print(f"[WARN] frame entry for {s} frame {fn} missing filepath and no match found; skipping frame.")
                if entries2:
                    category_data[s] = entries2
                    print(f"[INFO] Force-included sequence '{s}' with {len(entries2)} mapped frames from disk.")
                else:
                    print(f"[WARN] Could not map any frames for forced-included sequence '{s}'. Skipping.")
            else:
                print(f"[WARN] include_sequences requested '{s}' but no image files found under expected folders. Skipping.")

        # write output
        os.makedirs(output_dir, exist_ok=True)
        output_file = osp.join(output_dir, f"{category}_{subset}.jgz")
        with gzip.open(output_file, "w") as f:
            f.write(json.dumps(category_data).encode("utf-8"))

        print(f"[DONE] Wrote {output_file} (sequences: {len(category_data)})")



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.category == "all":
        categories = CATEGORIES
    else:
        categories = [args.category]

    # prepare include_sequences set (if provided)
    include_seqs = _load_include_sequences(args.include_sequences)

    for category in categories:
        process_poses(
            co3d_dir=args.co3d_v2_dir,
            category=category,
            output_dir=args.output_dir,
            min_quality=args.min_quality,
            subset_lists_file=args.subset_lists_file,
            ignore_quality=args.ignore_quality,
            include_sequences=include_seqs,
        )
