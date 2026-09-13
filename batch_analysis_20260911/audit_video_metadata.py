"""Read-only full-frame audit; outputs belong to the new batch audit directory."""
from pathlib import Path
import datetime
from fractions import Fraction
import hashlib
import json
import subprocess
import time

import cv2
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path('/home/fengbujue/项目/rppg识别')
BATCH = PROJECT / 'results/data1_6_20260911'
OUT = BATCH / 'reference_audit'
MANIFEST = BATCH / 'inputs_manifest.json'


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = ['expected_frames.json', 'video_metadata_audit.json', 'source_contact_sheet.jpg']
    assert not any((OUT / n).exists() for n in targets), 'Refusing to overwrite an existing audit'
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    cv2.setNumThreads(1)
    records = []
    thumbnails = {}
    print('START full sequential decode; data4 first; original video pixels, no inference resampling', flush=True)
    for item in sorted(manifest, key=lambda x: (x['case'] != 'data4', x['case'])):
        case = item['case']
        path = Path(item['video']['path'])
        assert path.resolve().is_relative_to((PROJECT / 'videos/data1_6_20260911').resolve())
        meta = item['ffprobe']['streams'][0]
        fps = Fraction(meta['avg_frame_rate'])
        started = time.monotonic()
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), f'{case} cannot open'
        header_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        header_fps = cap.get(cv2.CAP_PROP_FPS)
        # Keep just 3 small thumbnails; all frames are still decoded sequentially.
        indices = {round(float(f) * header_count): label for f, label in [(Fraction(1, 10), '10%'), (Fraction(1, 2), '50%'), (Fraction(9, 10), '90%')]}
        saved = {}
        count = 0
        dims = set()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dims.add((int(frame.shape[1]), int(frame.shape[0])))
            if count in indices:
                small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
                saved[indices[count]] = (count, cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            count += 1
        terminal_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.release()
        assert count > 0
        if count != header_count:
            # Contact sheet positions are based on actual decoded count, not an inconsistent header.
            saved = {}
            cap = cv2.VideoCapture(str(path))
            exact_targets = {round(f * count): label for f, label in [(0.1, '10%'), (0.5, '50%'), (0.9, '90%')]}
            for frame_index in range(max(exact_targets) + 1):
                ok, frame = cap.read()
                assert ok, f'{case} repeat decode ended unexpectedly'
                if frame_index in exact_targets:
                    small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
                    saved[exact_targets[frame_index]] = (frame_index, cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            cap.release()
        crosscheck = None
        if case == 'data4':
            cmd = ['ffprobe', '-v', 'error', '-threads', '1', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,r_frame_rate,time_base,duration,nb_frames,nb_read_frames', '-of', 'json', str(path)]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
            crosscheck = {'command_without_input_path': cmd[:-1], 'exit_code': p.returncode, 'stderr': p.stderr, 'result': json.loads(p.stdout) if p.returncode == 0 else None}
            assert p.returncode == 0
        duration = Fraction(count, 1) / fps
        duration_ns = round(duration * 10**9)
        mtime = int(item['video']['extended_unix_mtime_s'])
        rec = {
            'case': case, 'source_video': str(path), 'source_video_sha256_from_manifest': item['video']['sha256'],
            'source_hash_recomputed_here': False, 'source_bytes_match_manifest': path.stat().st_size == item['video']['bytes'],
            'method': 'OpenCV full sequential read of every original-resolution frame; no resized or temporal-subsampled input',
            'opencv_version': cv2.__version__, 'opencv_reported_frame_count': header_count,
            'opencv_reported_fps': header_fps, 'ffprobe_initial_metadata': meta,
            'decoded_frame_count': count, 'decoded_frame_dimensions': [list(x) for x in sorted(dims)],
            'terminal_opencv_frame_position': terminal_position, 'header_count_matches_decoded': header_count == count,
            'fps_numerator': fps.numerator, 'fps_denominator': fps.denominator,
            'duration_from_decoded_count_fraction_s': str(duration), 'duration_s': float(duration), 'duration_ns': duration_ns,
            'last_frame_nominal_time_s': float(Fraction(count - 1, 1) / fps),
            'original_zip_mtime_utc_s': mtime, 'conditional_video_start_utc_ns': mtime * 10**9 - duration_ns,
            'conditional_timing_note': 'Retains mtime-minus-N/fps assumption; decoding does not establish physical synchronization.',
            'ffprobe_full_count_crosscheck': crosscheck,
            'contact_sheet_positions': [{'fraction_label': label, 'frame_index_zero_based': index, 'nominal_time_s': float(Fraction(index, 1) / fps)} for label, (index, _) in saved.items()],
            'wall_time_s': time.monotonic() - started,
        }
        records.append(rec)
        thumbnails[case] = saved
        print(json.dumps({'case': case, 'decoded_count': count, 'opencv_header_count': header_count, 'fps': str(fps), 'duration_s': float(duration), 'ffprobe_nb_read_frames': crosscheck['result']['streams'][0].get('nb_read_frames') if crosscheck else None, 'wall_time_s': rec['wall_time_s']}, ensure_ascii=False), flush=True)
    records.sort(key=lambda x: x['case'])
    sheet = Image.new('RGB', (1496, 2002), '#edf1f5')
    draw = ImageDraw.Draw(sheet)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 19)
    small_font = ImageFont.truetype(font_path, 16)
    draw.text((18, 12), 'Original video contact sheet: 10% / 50% / 90% of decoded duration', font=font, fill='#172336')
    draw.text((18, 40), 'Original frames; thumbnails only resized for display. Three snapshots do not establish continuous motion quality.', font=small_font, fill='#475569')
    for row, rec in enumerate(records):
        y = 76 + row * 318
        draw.text((18, y), f"{rec['case']}  |  {rec['decoded_frame_dimensions'][0][0]}x{rec['decoded_frame_dimensions'][0][1]}  |  {float(Fraction(rec['fps_numerator'],rec['fps_denominator'])):.6f} fps  |  {rec['duration_s']:.3f} s", font=font, fill='#172336')
        for col, label in enumerate(['10%', '50%', '90%']):
            index, arr = thumbnails[rec['case']][label]
            x = 18 + col * 490
            sheet.paste(Image.fromarray(arr), (x, y + 29))
            t = float(Fraction(index * rec['fps_denominator'], rec['fps_numerator']))
            draw.rectangle((x, y + 270, x + 480, y + 299), fill='#172336')
            draw.text((x + 8, y + 274), f'{label} | frame {index} | t={t:.3f}s', font=small_font, fill='white')
    sheet.save(OUT / 'source_contact_sheet.jpg', quality=94, subsampling=0)
    write_json(OUT / 'expected_frames.json', {r['case']: r['decoded_frame_count'] for r in records})
    write_json(OUT / 'video_metadata_audit.json', {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'inputs_manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(), 'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': records, 'originals_modified': False, 'predictions_read': False, 'inference_rerun_or_resampled': False, 'contact_sheet_sha256': hashlib.sha256((OUT/'source_contact_sheet.jpg').read_bytes()).hexdigest(), 'limits': ['OpenCV stopping at EOF is cross-checked with ffprobe full frame counting for data4.', 'No scene identity inference is performed.', 'Counts validate decodable frames; synchronization and device HR delay remain uncalibrated.']})
    print('DONE '+str(OUT), flush=True)


if __name__ == '__main__':
    main()
