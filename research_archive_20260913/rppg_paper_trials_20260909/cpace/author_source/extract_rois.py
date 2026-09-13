"""
ROI Extraction Pipeline — MediaPipe FaceMesh
=============================================

Extracts per-frame mean RGB from 4 face ROIs and saves to CSV.
Run once per video; the output CSV is used by process_rois.py.

Output CSV format:
    Line 1: #fps=35.0,n_frames=2100,face_detect_rate=0.98
    Line 2: frame,Forehead_R,Forehead_G,Forehead_B,L_Cheek_R,...,Nose_B
    Lines 3+: per-frame data

Usage:
    # Single video
    python extract_rois.py <video_path> [--output <csv_path>]

    # Batch — UBFC-Phys (s1/vid_s1_T1.avi ...)
    python extract_rois.py --batch <ubfc_phys_dir> --tasks T1 T2 [--output <csv_dir>]

    # Batch — UBFC-rPPG (subject1/vid.avi ...)
    python extract_rois.py --batch-rppg <ubfc_rppg_dir> [--output <csv_dir>]

    # Batch — directory of video files
    python extract_rois.py --batch-videos <video_dir> [--output <csv_dir>]
"""

import numpy as np
import cv2
import csv
import os
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    import mediapipe as mp
except ImportError:
    print("Install mediapipe: pip install mediapipe")
    sys.exit(1)


# =========================================================================
# MEDIAPIPE FACE MESH ROI EXTRACTION
# =========================================================================

ROI_LANDMARKS = {
    # True forehead: upper face contour down to eyebrow line
    'Forehead': [10, 338, 297, 332, 284, 251,           # upper right
                 336, 296, 334, 293, 300,                 # right eyebrow
                 168,                                      # glabella center
                 107, 66, 105, 63, 70,                    # left eyebrow
                 21, 54, 103, 67, 109],                   # upper left
    'L_Cheek':  [36, 205, 206, 207, 187, 123, 116, 117,
                 118, 119, 120, 121, 128, 245, 193, 55],
    'R_Cheek':  [266, 425, 426, 427, 411, 352, 345, 346,
                 347, 348, 349, 350, 357, 465, 417, 285],
    'Nose':     [168, 6, 197, 195, 5, 4, 1, 19,          # midline: nasion to nose tip
                 94, 2, 164,                               # midline: just below tip
                 3, 51, 45,                                # left lateral bridge
                 248, 281, 275],                            # right lateral bridge
    # Full face oval boundary (MediaPipe FACEMESH_FACE_OVAL)
    'Full_Face': [10, 338, 297, 332, 284, 251, 389, 356,
                  454, 323, 361, 288, 397, 365, 379, 378,
                  400, 377, 152, 148, 176, 149, 150, 136,
                  172, 58, 132, 93, 234, 127, 162, 21,
                  54, 103, 67, 109],
}

ROI_NAMES = ['Forehead', 'L_Cheek', 'R_Cheek', 'Nose', 'Full_Face']


def _get_roi_mask(frame_shape, landmarks, indices):
    h, w = frame_shape[:2]
    points = []
    for idx in indices:
        lm = landmarks[idx]
        points.append([int(lm.x * w), int(lm.y * h)])
    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


def extract_and_save(video_path, csv_path, verbose=True):
    """Extract per-frame ROI RGB from video, save to CSV."""
    video_name = os.path.basename(video_path)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Extracting: {video_name}")
        print(f"{'='*60}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Collect per-frame data
    rows = []
    prev_masks = {}
    face_detected_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        row = {'frame': frame_idx}

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            face_detected_count += 1
            for roi_name in ROI_NAMES:
                indices = ROI_LANDMARKS[roi_name]
                mask = _get_roi_mask(frame.shape, landmarks, indices)
                prev_masks[roi_name] = mask
                pixels = frame[mask == 1]
                if len(pixels) > 0:
                    row[f'{roi_name}_B'] = float(pixels[:, 0].mean())
                    row[f'{roi_name}_G'] = float(pixels[:, 1].mean())
                    row[f'{roi_name}_R'] = float(pixels[:, 2].mean())
                else:
                    for ch in ['R', 'G', 'B']:
                        row[f'{roi_name}_{ch}'] = rows[-1][f'{roi_name}_{ch}'] if rows else 0.0
        else:
            for roi_name in ROI_NAMES:
                if roi_name in prev_masks:
                    pixels = frame[prev_masks[roi_name] == 1]
                    if len(pixels) > 0:
                        row[f'{roi_name}_B'] = float(pixels[:, 0].mean())
                        row[f'{roi_name}_G'] = float(pixels[:, 1].mean())
                        row[f'{roi_name}_R'] = float(pixels[:, 2].mean())
                    else:
                        for ch in ['R', 'G', 'B']:
                            row[f'{roi_name}_{ch}'] = rows[-1][f'{roi_name}_{ch}'] if rows else 0.0
                else:
                    for ch in ['R', 'G', 'B']:
                        row[f'{roi_name}_{ch}'] = 0.0

        rows.append(row)
        frame_idx += 1

        if verbose and frame_idx % 300 == 0:
            print(f"    Frames: {frame_idx}/{total_frames} "
                  f"({100 * frame_idx / max(total_frames, 1):.0f}%)")

    cap.release()
    face_mesh.close()

    n_frames = frame_idx
    face_detect_rate = face_detected_count / max(n_frames, 1)

    if verbose:
        print(f"  {n_frames} frames, {fps:.1f} FPS, {n_frames/fps:.1f}s")
        print(f"  Face detection: {face_detect_rate:.2%}")

    # Write CSV
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)

    fieldnames = ['frame']
    for roi in ROI_NAMES:
        fieldnames.extend([f'{roi}_R', f'{roi}_G', f'{roi}_B'])

    with open(csv_path, 'w', newline='') as f:
        # Metadata as comment line
        f.write(f"#fps={fps},n_frames={n_frames},face_detect_rate={face_detect_rate:.4f}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if verbose:
        print(f"  Saved: {csv_path}")

    return True


# =========================================================================
# BATCH MODES
# =========================================================================

def batch_ubfc_phys(ubfc_dir, tasks, output_dir, verbose=True):
    """Batch extract from UBFC-Phys directory structure."""
    subjects = []
    for entry in sorted(os.listdir(ubfc_dir)):
        subdir = os.path.join(ubfc_dir, entry)
        if os.path.isdir(subdir) and entry.startswith('s'):
            subjects.append(entry)
    subjects.sort(key=lambda x: int(x.replace('s', '')) if x.replace('s', '').isdigit() else 0)

    print(f"Found {len(subjects)} subjects, tasks: {tasks}")
    os.makedirs(output_dir, exist_ok=True)

    for subj in subjects:
        subj_dir = os.path.join(ubfc_dir, subj)
        subj_num = subj.replace('s', '')
        for task in tasks:
            # Find video
            video_path = None
            for vname in [f'vid_{subj}_{task}.avi', f'vid_s{subj_num}_{task}.avi',
                          f'vid_{subj}_{task}.mp4']:
                vp = os.path.join(subj_dir, vname)
                if os.path.exists(vp):
                    video_path = vp
                    break
            if video_path is None:
                if verbose:
                    print(f"  Skipping {subj}/{task}: no video found")
                continue

            csv_path = os.path.join(output_dir, f'{subj}_{task}.csv')
            if os.path.exists(csv_path):
                if verbose:
                    print(f"  Skipping {subj}/{task}: CSV already exists")
                continue

            try:
                extract_and_save(video_path, csv_path, verbose=verbose)
            except Exception as e:
                print(f"  ERROR {subj}/{task}: {e}")
                import traceback
                traceback.print_exc()


def batch_videos(video_dir, output_dir, verbose=True):
    """Batch extract from a flat directory of video files."""
    extensions = ('.mov', '.mp4', '.avi', '.mkv')
    videos = sorted([f for f in os.listdir(video_dir)
                     if f.lower().endswith(extensions)])
    print(f"Found {len(videos)} videos")
    os.makedirs(output_dir, exist_ok=True)

    for vname in videos:
        base = os.path.splitext(vname)[0]
        csv_path = os.path.join(output_dir, f'{base}.csv')
        if os.path.exists(csv_path):
            if verbose:
                print(f"  Skipping {vname}: CSV already exists")
            continue

        try:
            extract_and_save(os.path.join(video_dir, vname), csv_path, verbose=verbose)
        except Exception as e:
            print(f"  ERROR {vname}: {e}")
            import traceback
            traceback.print_exc()


def batch_ubfc_rppg(ubfc_rppg_dir, output_dir, verbose=True):
    """Batch extract from UBFC-rPPG directory structure.
    
    Expected structure:
        UBFC-rPPG/
            subject1/
                vid.avi
                ground_truth.txt
            subject2/
                vid.avi
                ground_truth.txt
            ...
    """
    subjects = []
    for entry in sorted(os.listdir(ubfc_rppg_dir)):
        subdir = os.path.join(ubfc_rppg_dir, entry)
        if os.path.isdir(subdir) and entry.startswith('subject'):
            subjects.append(entry)
    subjects.sort(key=lambda x: int(x.replace('subject', '')) if x.replace('subject', '').isdigit() else 0)

    print(f"Found {len(subjects)} subjects in UBFC-rPPG")
    os.makedirs(output_dir, exist_ok=True)

    for subj in subjects:
        subj_dir = os.path.join(ubfc_rppg_dir, subj)
        
        # Find video file
        video_path = None
        for vname in ['vid.avi', 'vid.mp4', 'video.avi', 'video.mp4']:
            vp = os.path.join(subj_dir, vname)
            if os.path.exists(vp):
                video_path = vp
                break
        
        if video_path is None:
            # Try any video file in the directory
            for f in os.listdir(subj_dir):
                if f.lower().endswith(('.avi', '.mp4', '.mkv', '.mov')):
                    video_path = os.path.join(subj_dir, f)
                    break
        
        if video_path is None:
            if verbose:
                print(f"  Skipping {subj}: no video found")
            continue

        csv_path = os.path.join(output_dir, f'{subj}.csv')
        if os.path.exists(csv_path):
            if verbose:
                print(f"  Skipping {subj}: CSV already exists")
            continue

        try:
            extract_and_save(video_path, csv_path, verbose=verbose)
        except Exception as e:
            print(f"  ERROR {subj}: {e}")
            import traceback
            traceback.print_exc()


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    output_dir = 'roi_csvs'
    tasks = ['T1', 'T2']
    mode = None
    video_path = None

    i = 0
    while i < len(args):
        if args[i] == '--batch':
            mode = 'ubfc_phys'
            video_path = args[i+1]
            i += 2
        elif args[i] == '--batch-rppg':
            mode = 'ubfc_rppg'
            video_path = args[i+1]
            i += 2
        elif args[i] == '--batch-videos':
            mode = 'flat'
            video_path = args[i+1]
            i += 2
        elif args[i] == '--output':
            output_dir = args[i+1]
            i += 2
        elif args[i] == '--tasks':
            tasks = []
            i += 1
            while i < len(args) and not args[i].startswith('--'):
                tasks.append(args[i])
                i += 1
        elif video_path is None:
            video_path = args[i]
            mode = 'single'
            i += 1
        else:
            i += 1

    if mode == 'ubfc_phys':
        batch_ubfc_phys(video_path, tasks, output_dir)
    elif mode == 'ubfc_rppg':
        batch_ubfc_rppg(video_path, output_dir)
    elif mode == 'flat':
        batch_videos(video_path, output_dir)
    elif mode == 'single':
        csv_out = output_dir if output_dir.endswith('.csv') else os.path.join(
            output_dir, os.path.splitext(os.path.basename(video_path))[0] + '.csv')
        extract_and_save(video_path, csv_out)
    else:
        print(__doc__)
