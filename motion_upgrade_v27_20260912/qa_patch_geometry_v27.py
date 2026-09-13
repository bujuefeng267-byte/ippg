"""Read-only independent geometry/quality-contract audit of V27 frontends.

Reconstructs local affine maps with QR, without calling the production anchor
projector. Does not rerun face detection, pixel tracking, HR, or reference
evaluation. Pixel quality is checked against recorded actual-pixel counts and
independently computed geometric visibility, not claimed as pulse quality.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import cv2
import numpy as np
import pandas as pd


REGION_BOXES = {'forehead': (.32, .12, .68, .30),
                'left_cheek': (.16, .46, .40, .70),
                'right_cheek': (.60, .46, .84, .70)}
PATCH_IDS = tuple(f'{region}_{i}' for region in REGION_BOXES for i in range(4))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def boolean(values):
    a = np.asarray(values)
    assert not pd.isna(a).any() and np.isin(a, [0, 1, False, True]).all()
    return a.astype(bool)


def audit_arrays(frames, patches, points, anchors, config, fps, image_shape):
    n = len(frames)
    height, width = image_shape
    assert points.ndim == 3 and points.shape[0] == n and points.shape[2] == 2
    np.testing.assert_array_equal(frames.frame, np.arange(n))
    np.testing.assert_allclose(frames.time_s, np.arange(n)/fps, atol=1e-8, rtol=0)
    assert len(patches) == 12*n and not patches[['frame', 'patch_id']].duplicated().any()
    assert set(patches.patch_id) == set(PATCH_IDS)
    missing = frames.source.to_numpy() == 'missing'
    detected = frames.source.isin(['mesh', 'redetected']).to_numpy()
    bridge = frames.source.to_numpy() == 'flow_tracked'
    assert (missing | detected | bridge).all()
    np.testing.assert_array_equal(boolean(frames.face_detected), detected)
    assert np.isnan(points[missing]).all()
    if (~missing).any():
        assert points.shape[1] >= 3 and np.isfinite(points[~missing]).all()
    assert (frames.bridge_age_frames.to_numpy()[bridge] >= 1).all()
    assert (frames.bridge_age_frames.to_numpy()[bridge] <= round(config['max_bridge_s']*fps)).all()
    assert (frames.bridge_age_frames.to_numpy()[detected] == 0).all()
    if 'is_bridge' in frames:
        np.testing.assert_array_equal(boolean(frames.is_bridge), bridge)
    bounds = np.full((n, 4), np.nan)
    if (~missing).any():
        percentiles = np.percentile(points[~missing], [2, 98], axis=1)
        bounds[~missing, :2] = percentiles[0]
        bounds[~missing, 2:] = percentiles[1]
    np.testing.assert_allclose(frames[['face_x0', 'face_y0', 'face_x1', 'face_y1']],
                               bounds/[width, height, width, height], atol=1e-9, rtol=0, equal_nan=True)
    usable_face = ~missing & ((bounds[:, 2:]-bounds[:, :2]) >= config['min_face_pixels']).all(1)
    initial = anchors['initial_frame']
    if anchors['patches'] is not None:
        assert isinstance(initial, int) and 0 <= initial < n and detected[initial]
        assert set(anchors['patches']) == set(PATCH_IDS)
        assert usable_face[initial]
    else:
        assert initial is None
    details = []
    viewport = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    for ident in PATCH_IDS:
        table = patches.loc[patches.patch_id == ident].sort_values('frame')
        np.testing.assert_array_equal(table.frame, np.arange(n))
        np.testing.assert_allclose(table.time_s, frames.time_s, atol=1e-8, rtol=0)
        region, index = ident.rsplit('_', 1)
        index = int(index)
        assert (table.region == region).all()
        np.testing.assert_array_equal(table.source, frames.source)
        np.testing.assert_array_equal(boolean(table.face_detected), detected)
        np.testing.assert_array_equal(boolean(table.is_bridge), bridge)
        np.testing.assert_array_equal(table.bridge_age_frames, frames.bridge_age_frames)
        expected_quad = np.full((n, 4, 2), np.nan)
        residual = np.full(n, np.nan)
        status = np.full(n, 'anchors_unavailable', dtype=object)
        if anchors['patches'] is not None:
            item = anchors['patches'][ident]
            assert item['region'] == region and item['row'] == index//2 and item['column'] == index%2
            ids = np.asarray(item['landmark_indices'])
            assert ids.shape == (config['local_landmarks'],) and np.issubdtype(ids.dtype, np.integer)
            assert len(set(ids.tolist())) == len(ids) and (ids >= 0).all() and (ids < points.shape[1]).all()
            x0, y0, x1, y1 = bounds[initial]
            fw, fh = x1-x0, y1-y0
            rx0, ry0, rx1, ry1 = REGION_BOXES[region]
            cw, ch = (rx1-rx0)*fw/2, (ry1-ry0)*fh/2
            left, top = x0+rx0*fw+(index%2)*cw, y0+ry0*fh+(index//2)*ch
            inset = config['cell_inset_fraction']
            first_quad = np.array([[left+inset*cw, top+inset*ch],
                                   [left+(1-inset)*cw, top+inset*ch],
                                   [left+(1-inset)*cw, top+(1-inset)*ch],
                                   [left+inset*cw, top+(1-inset)*ch]])
            center = first_quad.mean(0)
            expected_ids = np.argsort(np.sum(((points[initial]-center)/fh)**2, axis=1),
                                     kind='stable')[:config['local_landmarks']]
            np.testing.assert_array_equal(ids, expected_ids)
            np.testing.assert_allclose(item['initial_quad_px'], first_quad, atol=1e-9, rtol=0)
            local = (points[initial, ids]-center)/fh
            local_quad = (first_quad-center)/fh
            np.testing.assert_allclose(item['local_landmark_coordinates'], local, atol=1e-12, rtol=0)
            np.testing.assert_allclose(item['local_quad_coordinates'], local_quad, atol=1e-12, rtol=0)
            design = np.column_stack([local, np.ones(len(ids))])
            # Independent QR solve; production uses per-frame lstsq/SVD.
            q, r = np.linalg.qr(design, mode='reduced')
            weights = np.linalg.solve(r, q.T)
            active = usable_face & (np.arange(n) >= initial)
            indices = np.flatnonzero(active)
            status[np.arange(n) >= initial] = 'invalid_face_geometry'
            status[missing & (np.arange(n) >= initial)] = 'landmarks_missing'
            if len(indices):
                selected = points[indices][:, ids, :]
                mappings = np.einsum('ij,njk->nik', weights, selected)
                fit = np.einsum('ij,njk->nik', design, mappings)
                residual[indices] = np.sqrt(np.mean(np.sum((fit-selected)**2, axis=2), axis=1))/(bounds[indices, 3]-bounds[indices, 1])
                axes = np.linalg.svd(mappings[:, :2], compute_uv=False)
                good = (np.isfinite(mappings).all(axis=(1, 2)) & (axes[:, -1] > 1e-8) &
                        (axes[:, 0]/np.maximum(axes[:, -1], 1e-30) <= config['max_local_axis_ratio']) &
                        (np.linalg.det(mappings[:, :2]) > 0) &
                        (residual[indices] <= config['max_anchor_residual_face_fraction']))
                status[indices] = 'local_anchor_geometry_rejected'
                status[indices[good]] = 'anchored'
                corners = np.einsum('ij,njk->nik', np.column_stack([local_quad, np.ones(4)]), mappings)
                expected_quad[indices[good]] = corners[good]
        observed_quad = table[[f'corner{i}_{axis}' for i in range(4) for axis in 'xy']].to_numpy(float).reshape(n, 4, 2)
        np.testing.assert_allclose(observed_quad, expected_quad, atol=1e-7, rtol=0, equal_nan=True)
        expected_geometry = np.isfinite(expected_quad).all(axis=(1, 2))
        np.testing.assert_array_equal(boolean(table.geometry_valid), expected_geometry)
        np.testing.assert_array_equal(table.geometry_status.to_numpy(), status)
        np.testing.assert_allclose(table.anchor_residual_face_fraction, residual, atol=1e-9, rtol=0, equal_nan=True)
        visible = np.zeros(n)
        for i in np.flatnonzero(expected_geometry):
            quad = expected_quad[i].astype(np.float32)
            area = cv2.contourArea(quad)
            intersection, _ = cv2.intersectConvexConvex(quad, viewport)
            visible[i] = np.clip(intersection/area, 0, 1) if area > 0 else 0
        np.testing.assert_allclose(table.visible_fraction, visible, atol=1e-6, rtol=0)
        count, valid_count = table.pixel_count.to_numpy(float), table.valid_pixels.to_numpy(float)
        assert np.isfinite(count).all() and np.isfinite(valid_count).all()
        assert ((count >= 0) & (count == np.floor(count))).all()
        assert ((valid_count >= 0) & (valid_count == np.floor(valid_count)) & (valid_count <= count)).all()
        assert (count[~expected_geometry] == 0).all()
        valid = boolean(table.valid)
        np.testing.assert_array_equal(valid, valid_count >= config['min_valid_pixels'])
        fraction = np.divide(valid_count, count, out=np.zeros(n), where=count > 0)
        np.testing.assert_allclose(table.valid_pixel_fraction, fraction, atol=1e-12, rtol=0)
        np.testing.assert_allclose(table.quality, np.where(valid, fraction*table.visible_fraction, 0.), atol=1e-12, rtol=0)
        for prefix in ('raw', 'tracked'):
            values = table[[f'{prefix}_{c}' for c in 'rgb']].to_numpy(float)
            assert np.isnan(values[~valid]).all() and np.isfinite(values[valid]).all()
        np.testing.assert_array_equal(table.pixel_source.to_numpy() == 'missing', ~valid)
        finite = np.isfinite(expected_quad)
        details.append(dict(patch_id=ident, region=region, frames=n, valid_frames=int(valid.sum()),
            projected_frames=int(expected_geometry.sum()),
            maximum_corner_difference_px=float(abs(observed_quad[finite]-expected_quad[finite]).max()) if finite.any() else 0.,
            fixed_anchor_IDs_and_local_offsets_verified=True, geometry_and_missing_masks_exact=True,
            independent_visible_area_verified=True, reported_pixel_quality_contract_verified=True))
    return dict(frames=n, patches=details, stable_patch_count=12, parent_region_count=3,
                global_bbox_clock_and_bridge_verified=True,
                geometry_projection_method='independent vectorized QR solve; no production projector called',
                quality_limitation='Uses recorded measured-pixel counts; does not independently remeasure raw RGB or run pixel tracking.',
                no_physiological_quality_or_accuracy_claim=True)


def audit_frontend(front):
    front = Path(front).resolve()
    meta = json.loads((front/'metadata.json').read_text())
    assert meta['reference_used'] is False and meta['anchors_reinitialized_after_gap'] is False
    assert meta['temporal_RGB_filling'] is False and meta['RGB_aggregation_across_patches'] is False
    files = {str(front/'metadata.json'): sha(front/'metadata.json')}
    for name, expected in meta['output_hashes'].items():
        assert name in ('patch_trace.csv', 'frame_trace.csv', 'patch_anchors.json', 'landmark_trace.npz')
        path = front/name
        assert sha(path) == expected
        files[str(path)] = expected
    here = Path(__file__).resolve().parent
    for name, expected in meta['source_hashes'].items():
        assert sha(here/name) == expected
        files[str(here/name)] = expected
    video = Path(meta['video_path'])
    assert sha(video) == meta['video_sha256'] and video.stat().st_size == meta['video_bytes']
    files[str(video)] = meta['video_sha256']
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        cap.release()
    assert min(shape) > 0 and np.isclose(fps, meta['fps'], atol=1e-6, rtol=0)
    frames, patches = pd.read_csv(front/'frame_trace.csv'), pd.read_csv(front/'patch_trace.csv')
    assert len(frames) == meta['frames']
    anchors = json.loads((front/'patch_anchors.json').read_text())
    with np.load(front/'landmark_trace.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['frame'], frames.frame)
        np.testing.assert_allclose(archive['time_s'], frames.time_s, atol=1e-8, rtol=0)
        points = archive['points_px']
    result = audit_arrays(frames, patches, points, anchors, meta['config'], fps, shape)
    for path, expected in files.items():
        assert sha(path) == expected, 'Input changed during audit: '+path
    return dict(frontend=str(front), fps=fps, source_files_unchanged=True, source_hashes=files, **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve existing receipts'
    results = []
    for front in args.frontend:
        results.append(audit_frontend(front))
        print(json.dumps(dict(frontend=str(front), passed=True, frames=results[-1]['frames'])), flush=True)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True, errors=[],
                   reference_files_read=False, pixel_tracking_rerun=False, QA_source_sha256=sha(__file__), cases=results)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(str(args.output), flush=True)


if __name__ == '__main__':
    main()
