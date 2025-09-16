#!/usr/bin/env python3
"""
test_co3d_depth.py (patched)

 python .\test_co3d_depth.py 
 --co3d_dir F:\Datasets\Blind_images_v2\ 
 --co3d_anno_dir . 
 --model_path .\model_tracker_fixed_e20.pt 
 --category backpack 
 --sequence 159_17477_32932 
 --gt_depth_dir F:\Datasets\Blind_images_v2\backpack\159_17477_32932\depths\ 
 --gt_mask_dir F:\Datasets\Blind_images_v2\backpack\159_17477_32932\depth_masks\ 
 --dump_preds_shapes 

CO3Dv2 depth evaluation using VGGT model predictions.

This patched version adds:
 - diagnostics (per-frame and global stats)
 - automatic unit checks and mm<->m correction heuristics
 - detection/handling of model outputs being in camera-frame vs world-frame
 - robust extraction of per-frame extrinsics/intrinsics from pose_enc or annotation
 - updated torch.amp.autocast usage (no FutureWarning)
"""

import argparse
import gzip
import json
import os
import os.path as osp
import random
import numpy as np
import torch
from PIL import Image
import cv2
from scipy.spatial import cKDTree

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# ---------- utils ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--co3d_dir', required=True)
    p.add_argument('--co3d_anno_dir', required=None, default='.')
    p.add_argument('--model_path', required=None, default='./model_tracker_fixed_e20.pt')
    p.add_argument('--category', required=True)
    p.add_argument('--sequence', default=None)
    p.add_argument('--num_frames', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--min_num_images', type=int, default=20)
    p.add_argument('--depth_conf_thresh', type=float, default=0.2)
    p.add_argument('--max_points', type=int, default=100000)
    p.add_argument('--gt_depth_dir', type=str, default=None)
    p.add_argument('--gt_mask_dir', type=str, default=None)
    p.add_argument('--dump_preds_shapes', action='store_true')
    p.add_argument('--resize_interp', choices=['nearest','linear'], default='nearest')
    p.add_argument('--verbose', action='store_true', help='Show debug diagnostics (STATS/DBG/DIAG etc.)')
    return p.parse_args()

def load_annotation(anno_file):
    with gzip.open(anno_file, 'r') as f:
        return json.loads(f.read())

def umeyama_align(src, dst, with_scaling=True):
    src = np.asarray(src).astype(np.float64)
    dst = np.asarray(dst).astype(np.float64)
    assert src.shape[1] == 3 and dst.shape[1] == 3
    N = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / N
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2,2] = -1
    R = U @ S @ Vt
    if with_scaling:
        var_src = (src_c ** 2).sum() / N
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return R, t, s

def chamfer_metrics(pred_pts, gt_pts):
    if len(pred_pts)==0 or len(gt_pts)==0:
        return np.inf, np.inf, np.inf
    tree_gt = cKDTree(gt_pts)
    d_pred_to_gt, _ = tree_gt.query(pred_pts, k=1)
    tree_pred = cKDTree(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts, k=1)
    acc = float(d_pred_to_gt.mean())
    comp = float(d_gt_to_pred.mean())
    chamfer = 0.5*(acc + comp)
    return acc, comp, chamfer

# ---------- intrinsics helpers ----------
def build_intrinsic_matrix(focal_length, principal_point):
    # focal_length may be scalar or [fx,fy]; principal_point [cx,cy]
    if isinstance(focal_length, (list, tuple, np.ndarray)):
        fx = float(focal_length[0])
        fy = float(focal_length[1]) if len(focal_length) > 1 else float(focal_length[0])
    else:
        fx = fy = float(focal_length)
    cx, cy = float(principal_point[0]), float(principal_point[1])
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    return K

def resize_depth_and_scale_intrinsics(depth, K_orig, target_h, target_w, interp='nearest'):
    h0,w0 = depth.shape
    if (h0,w0) == (target_h, target_w):
        return depth, (K_orig.copy() if K_orig is not None else None)
    sx = target_w / float(w0)
    sy = target_h / float(h0)
    if interp == 'nearest':
        depth_r = cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        depth_r = cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    depth_r = depth_r.astype(np.float64)
    if K_orig is None:
        return depth_r, None
    K_scaled = K_orig.copy().astype(np.float64)
    K_scaled[0,0] *= sx
    K_scaled[1,1] *= sy
    K_scaled[0,2] *= sx
    K_scaled[1,2] *= sy
    return depth_r, K_scaled

def try_load_image_depth(path, debug=False):
    if not osp.exists(path):
        raise FileNotFoundError(path)
    ext = osp.splitext(path)[1].lower()
    if ext in ['.npy', '.npz']:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.keys())[0]]
        arr = np.asarray(arr)
    else:
        im = Image.open(path)
        arr = np.array(im)
    if debug:
        try:
            amin = float(arr.min()); amax = float(arr.max())
        except Exception:
            amin = None; amax = None
        print(f"[DBG] loaded depth file {path} -> shape={arr.shape}, dtype={arr.dtype}, min={amin}, max={amax}")
    # common encodings:
    if arr.ndim == 3 and arr.dtype == np.uint8 and arr.shape[2] >= 3:
        R = arr[...,0].astype(np.uint32)
        G = arr[...,1].astype(np.uint32)
        B = arr[...,2].astype(np.uint32)
        depth_int = R + (G << 8) + (B << 16)
        depth = depth_int.astype(np.float64)
        if depth.max() > 10000:
            depth = depth / 1000.0
        return depth
    if arr.ndim == 3:
        counts = [np.count_nonzero(arr[...,c]) for c in range(arr.shape[2])]
        best = int(np.argmax(counts))
        depth = arr[...,best].astype(np.float64)
    else:
        depth = arr.astype(np.float64)
    if np.nanmax(depth) > 10000:
        depth = depth / 1000.0
    return depth

# ---------- numpy unprojection ----------
def unproject_depth_numpy(depth, K, extri, mask=None):
    """
    depth: (H,W) depth in meters
    K: 3x3 intrinsics (fx,fy,cx,cy). If None -> cannot compute -> returns empty
    extri: 3x4 extrinsic (w2c) (camera from world). If None -> returns cam coords without world transform
    mask: optional boolean mask (H,W) True = keep
    Returns points_world (N,3) in world coordinates (if extri provided) else camera coords.
    """
    if K is None:
        raise RuntimeError("Intrinsic K is required to unproject with this function.")
    H,W = depth.shape
    ys, xs = np.indices((H,W))
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
    z = depth
    valid = (z > 0) & (~np.isnan(z))
    if mask is not None:
        mask = cv2.resize(mask.astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST).astype(bool)
        valid = valid & mask
    if valid.sum() == 0:
        return np.zeros((0,3), dtype=np.float64)
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    Z = z
    pts_cam = np.stack([X,Y,Z], axis=-1).reshape(-1,3)
    valid_flat = valid.reshape(-1)
    pts_cam = pts_cam[valid_flat]
    # transform to world if extri provided (extri is w2c: camera from world)
    if extri is not None:
        R = extri[:3,:3]
        t = extri[:3,3]
        Rinv = R.T
        pts_world = (Rinv @ (pts_cam.T - t[:,None])).T
        return pts_world
    else:
        return pts_cam

def tensor_frame_to_map(tensor, frame_idx, batch_size):
    if tensor is None:
        return None
    arr = tensor.detach().cpu().numpy()
    # common cases
    if arr.ndim == 5:
        # (B,S,H,W,C) or (1,S,H,W,1)
        if arr.shape[0] == 1:
            arr = arr[0]
    if arr.ndim == 4:
        # (S,H,W,C) or (B,S,H,W) or (B,H,W,C)
        if arr.shape[0] == batch_size:
            return np.squeeze(arr[frame_idx])
        if arr.shape[0] == 1:
            return np.squeeze(arr[0])
        if arr.shape[0] > frame_idx:
            return np.squeeze(arr[frame_idx])
        return np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[0] == batch_size:
            return np.squeeze(arr[frame_idx])
        if batch_size == 1 and arr.shape[0] == 1:
            return np.squeeze(arr[0])
        return np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    return np.squeeze(arr)

# ---------- helpers for tensors/extrinsics ----------
def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def get_extrinsic_for_frame(extrinsic_all, fi):
    """
    Extract a 3x4 extrinsic matrix for frame fi from extrinsic_all which may be:
     - torch tensor with shape (1, S, 3, 4) or (S,3,4) or (1,S,12) etc.
     - numpy array with similar shapes.
    Returns None if cannot extract.
    """
    if extrinsic_all is None:
        return None
    arr = to_numpy(extrinsic_all)
    if arr is None:
        return None
    # normalize
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    # possible shapes now: (S,3,4) or (S,12) or (3,4) or (1,3,4)
    if arr.ndim == 3 and arr.shape[0] > fi and arr.shape[1] == 3 and arr.shape[2] == 4:
        return arr[fi]
    if arr.ndim == 2 and arr.size == 12:
        # single extrinsic flattened
        if fi == 0:
            return arr.reshape(3,4)
        else:
            return None
    if arr.ndim == 3 and arr.shape[1] == 3 and arr.shape[2] == 4:
        # fallback for safety
        try:
            return arr[fi]
        except Exception:
            return None
    # unsupported shape
    return None

def cam_to_world(pts_cam, extri):
    """Convert pts in camera coords to world coords using extri (w2c)."""
    if extri is None:
        return pts_cam
    R = extri[:3,:3]
    t = extri[:3,3]
    pts_world = (R.T @ (pts_cam.T - t[:,None])).T
    return pts_world

# small diagnostics helpers
def stats_info(name, arr):
    if arr is None or arr.size == 0:
        print(f"  [STATS] {name}: EMPTY")
        return
    arr_f = arr.reshape(-1,3)
    med = np.median(np.abs(arr_f), axis=0)
    mean = np.mean(arr_f, axis=0)
    mn = np.min(arr_f, axis=0)
    mx = np.max(arr_f, axis=0)
    print(f"  [STATS] {name}: n={arr_f.shape[0]} mean={mean} med_abs={med} min={mn} max={mx}")

def bbox_diag(pts):
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return np.linalg.norm(mx - mn)

def sample_pts(pts, n=20000):
    if pts.shape[0] <= n:
        return pts
    idx = np.random.choice(pts.shape[0], n, replace=False)
    return pts[idx]

# ---------- main ----------
def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = args.device
    if device.startswith('cuda'):
        cc = torch.cuda.get_device_capability(device=None)[0]
        dtype = torch.bfloat16 if cc >= 8 else torch.float16
    else:
        dtype = torch.float32

    print("Loading model:", args.model_path)
    if args.model_path.startswith("facebook/") and not osp.exists(args.model_path):
        model = VGGT.from_pretrained(args.model_path).to(device)
    else:
        model = VGGT()
        state = torch.load(args.model_path, map_location='cpu')
        try:
            model.load_state_dict(state)
        except Exception:
            model.load_state_dict(state, strict=False)
        model.to(device)
    model.eval()

    anno_file = osp.join(args.co3d_anno_dir, f"{args.category}_test.jgz")
    if not osp.exists(anno_file):
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")
    annotation = load_annotation(anno_file)

    seq_names = sorted(list(annotation.keys()))
    if args.sequence is not None:
        if args.sequence not in annotation:
            raise ValueError(f"Sequence {args.sequence} not found in annotation.")
        seq_names = [args.sequence]

    if args.gt_depth_dir is None:
        print("[WARN] No GT depth dir provided (--gt_depth_dir). Exiting.")
        return

    results = {}

    for seq_name in seq_names:
        print("Processing", seq_name)
        seq_data = annotation[seq_name]
        if len(seq_data) < args.min_num_images:
            print(f" skip (only {len(seq_data)} images < min {args.min_num_images})")
            continue

        # select frames
        if len(seq_data) <= args.num_frames:
            chosen = list(range(len(seq_data)))
        else:
            chosen = sorted(np.random.choice(len(seq_data), args.num_frames, replace=False).tolist())
        print(" chosen indices:", chosen)

        # build lists of (filepath,entry)
        image_paths = []
        entries = []
        for idx in chosen:
            entry = seq_data[idx]
            img_path = osp.normpath(osp.join(args.co3d_dir, entry['filepath']))
            image_paths.append(img_path)
            entries.append(entry)

        valid_pairs = [(p,e) for p,e in zip(image_paths, entries) if osp.exists(p)]
        if len(valid_pairs) == 0:
            print(" no valid images for sequence -> skipping")
            continue
        image_paths, entries = zip(*valid_pairs)

        # load images to get model input size
        imgs_tensor = load_and_preprocess_images(list(image_paths)).to(device)
        target_h = imgs_tensor.shape[-2]; target_w = imgs_tensor.shape[-1]
        print(f" model image size: H={target_h}, W={target_w}")

        # --- build GT point cloud by unprojecting GT depths (resized to model size) ---
        gt_points_list = []
        for i, idx in enumerate(chosen):
            entry = seq_data[idx]
            basename = os.path.basename(entry['filepath'])
            name_wo_ext = osp.splitext(basename)[0]
            # try a few filenames in depth dir
            cand = [
                osp.join(args.gt_depth_dir, f"{basename}.geometric.png"),
                osp.join(args.gt_depth_dir, f"{name_wo_ext}.jpg.geometric.png"),
                osp.join(args.gt_depth_dir, f"{name_wo_ext}.geometric.png"),
                osp.join(args.gt_depth_dir, f"{name_wo_ext}.png"),
                osp.join(args.gt_depth_dir, basename),
            ]
            depth_path = None
            for p in cand:
                if osp.exists(p):
                    depth_path = p; break
            if depth_path is None:
                print(f"  [WARN] no GT depth for {basename}, tried {cand[:3]}")
                continue
            try:
                depth_raw = try_load_image_depth(depth_path, debug=args.verbose)
            except Exception as e:
                print(f"  [WARN] cannot load GT depth {depth_path}: {e}")
                continue

            # intrinsics from entry if available
            K = None
            if 'focal_length' in entry and 'principal_point' in entry:
                try:
                    K = build_intrinsic_matrix(entry['focal_length'], entry['principal_point'])
                except Exception:
                    K = None

            depth_r, K_scaled = resize_depth_and_scale_intrinsics(depth_raw, K, target_h, target_w, interp=args.resize_interp)
            # mask if provided
            mask = None
            if args.gt_mask_dir:
                m_candidates = [osp.join(args.gt_mask_dir, f"{name_wo_ext}.png"), osp.join(args.gt_mask_dir, f"{basename}.png")]
                for mp in m_candidates:
                    if osp.exists(mp):
                        mask = np.array(Image.open(mp).convert('L'))>0
                        break

            # extrinsic from annotation (w2c)
            extri = None
            if 'extri' in entry:
                extri = np.array(entry['extri'])
            elif 'R' in entry and 'T' in entry:
                Rpt = np.array(entry['R']); Tpt = np.array(entry['T'])
                Tpt[:2] *= -1; Rpt[:, :2] *= -1; Rpt = Rpt.transpose(1,0)
                extri = np.hstack((Rpt, Tpt[:,None]))
            else:
                print(f"  [WARN] no extrinsic for frame {idx} -> cannot unproject GT")
                continue

            if K_scaled is None:
                print(f"  [WARN] no intrinsics for frame {idx} -> cannot unproject GT")
                continue
            try:
                pts_world = unproject_depth_numpy(depth_r, K_scaled, extri, mask=mask)
                if pts_world.shape[0] > 0:
                    gt_points_list.append(pts_world.astype(np.float64))
            except Exception as e:
                print(f"  [WARN] failed unproject GT {depth_path}: {e}")
                continue

        if len(gt_points_list) == 0:
            print("  [WARN] built empty GT point cloud -> skipping sequence")
            continue
        gt_pts = np.concatenate(gt_points_list, axis=0)
        print(f" GT points built: {gt_pts.shape[0]} pts")
        #stats_info("GT_sample_before", gt_pts[:1000])
        if args.verbose:
            stats_info("GT_sample_before", gt_pts[:1000])

        # --- run model on images ---
        with torch.no_grad():
            if device.startswith('cuda'):
                with torch.amp.autocast(device_type='cuda', dtype=dtype):
                    preds = model(imgs_tensor)
            else:
                preds = model(imgs_tensor)

        if args.dump_preds_shapes:
            print("PRED KEYS & SHAPES:")
            if isinstance(preds, dict):
                for k,v in preds.items():
                    try:
                        print(f"  {k}: {getattr(v,'shape',None)} {getattr(v,'dtype',None)}")
                    except Exception:
                        print(f"  {k}: type {type(v)}")
            else:
                print(" preds type:", type(preds))

        # --- collect predicted world points (improved with diagnostics & heuristics) ---
        pred_world_list = []

        # try to extract extrinsic/intrinsic from pose_enc for possible camera->world transform
        extrinsic_all = None; intrinsic_all = None
        if isinstance(preds, dict) and 'pose_enc' in preds:
            try:
                extrinsic_all, intrinsic_all = pose_encoding_to_extri_intri(preds['pose_enc'], imgs_tensor.shape[-2:])
            except Exception as e:
                print("  [WARN] pose_encoding_to_extri_intri failed (ignored):", e)
                extrinsic_all, intrinsic_all = None, None

        # If model directly outputs world_points, prefer that (but check if it's camera-frame)
        if isinstance(preds, dict) and 'world_points' in preds:
            wp = preds['world_points']  # shape examples: (1,S,H,W,3) or (1,S,3,H,W) etc.
            wp_conf = preds.get('world_points_conf', None)
            for fi in range(len(entries)):
                arr = tensor_frame_to_map(wp, fi, imgs_tensor.shape[0])
                if arr is None: continue
                arr = np.asarray(arr)
                # normalize shapes to (H,W,3) or (3,H,W)
                if arr.ndim == 4 and arr.shape[-1] == 3:
                    pts = arr.reshape(-1,3)
                elif arr.ndim == 3 and arr.shape[0] == 3:
                    pts = arr.reshape(3,-1).T
                elif arr.ndim == 3 and arr.shape[2] == 3:
                    pts = arr.reshape(-1,3)
                else:
                    try:
                        pts = arr.reshape(-1,3)
                    except Exception:
                        print(f"   [WARN] unexpected world_points frame shape: {arr.shape}")
                        continue

                # apply per-point confidence mask if available
                if wp_conf is not None:
                    conf = tensor_frame_to_map(wp_conf, fi, imgs_tensor.shape[0])
                    if conf is not None:
                        cf = np.asarray(conf).reshape(-1)
                        minlen = min(cf.shape[0], pts.shape[0])
                        pts = pts[:minlen][cf[:minlen] > 0.0]

                if pts.shape[0] == 0:
                    continue

                pts = pts.astype(np.float64)
                if args.verbose and fi < 3:
                    stats_info(f"pred_frame_{fi}_raw", pts)
                pred_world_list.append(pts)
        else:
            # fallback: unproject predicted depth maps (existing logic, preserved)
            depth_tensor = None; depth_conf_tensor = None
            if isinstance(preds, dict):
                for k in ['depth','depth_map','pred_depth','dmap']:
                    if k in preds:
                        depth_tensor = preds[k]; break
                for k in ['depth_conf','depth_confidence','dconf']:
                    if k in preds:
                        depth_conf_tensor = preds[k]; break

            if depth_tensor is None:
                print("  [WARN] no predicted depth available from model -> skipping predictions")
            else:
                for fi, entry in enumerate(entries):
                    frame_idx = int(chosen[fi])
                    # extrinsic
                    extri = None
                    if extrinsic_all is not None:
                        try:
                            extri = get_extrinsic_for_frame(extrinsic_all, fi)
                        except Exception:
                            extri = None
                    if extri is None and 'extri' in entry:
                        extri = np.array(entry['extri'])
                    elif extri is None and 'R' in entry and 'T' in entry:
                        Rpt = np.array(entry['R']); Tpt = np.array(entry['T'])
                        Tpt[:2] *= -1; Rpt[:, :2] *= -1; Rpt = Rpt.transpose(1,0)
                        extri = np.hstack((Rpt, Tpt[:,None]))
                    if extri is None:
                        print(f"   [WARN] no extrinsic for predicted frame {frame_idx} -> skip")
                        continue

                    # intrinsics
                    intr = None
                    if intrinsic_all is not None:
                        try:
                            intr = get_extrinsic_for_frame(intrinsic_all, fi)  # reuse helper
                        except Exception:
                            intr = None
                    if intr is None and 'focal_length' in entry and 'principal_point' in entry:
                        intr = build_intrinsic_matrix(entry['focal_length'], entry['principal_point'])
                    if intr is None:
                        print(f"   [WARN] no intrinsics for frame {frame_idx} -> skip")
                        continue

                    # get depth numpy
                    try:
                        d_np = tensor_frame_to_map(depth_tensor, fi, imgs_tensor.shape[0])
                        d_np = np.asarray(d_np, dtype=np.float64)
                    except Exception as e:
                        print(f"   [WARN] cannot extract predicted depth for frame {frame_idx}: {e}")
                        continue

                    # squeeze trailing singleton channels
                    if d_np.ndim == 3 and d_np.shape[-1] == 1:
                        d_np = np.squeeze(d_np, axis=-1)
                    if d_np.ndim == 4 and d_np.shape[0] == 1:
                        d_np = np.squeeze(d_np, axis=0)

                    # predicted depth should already be model size, but if not, resize
                    if d_np.shape != (target_h, target_w):
                        d_np = cv2.resize(d_np.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(np.float64)

                    # depth confidence
                    if depth_conf_tensor is not None:
                        cd_np = tensor_frame_to_map(depth_conf_tensor, fi, imgs_tensor.shape[0])
                        cd_np = np.asarray(cd_np, dtype=np.float64)
                        if cd_np.ndim==3 and cd_np.shape[-1]==1:
                            cd_np = np.squeeze(cd_np, axis=-1)
                    else:
                        cd_np = np.ones_like(d_np)

                    # reconcile shapes
                    if cd_np.shape != d_np.shape:
                        cd_np = np.ones_like(d_np)

                    mask_valid = (cd_np >= args.depth_conf_thresh) & (d_np > 0) & (~np.isnan(d_np))
                    if mask_valid.sum() == 0:
                        print(f"   [WARN] no valid predicted depth pixels after confidence mask for frame {frame_idx}")
                        continue

                    # scale intrinsics if intr corresponded to different resolution: use 'width'/'height' if available
                    K = intr.copy()
                    if 'width' in entry and 'height' in entry:
                        ow = float(entry['width']); oh = float(entry['height'])
                        sx = target_w / ow; sy = target_h / oh
                        K[0,0] *= sx; K[1,1] *= sy; K[0,2] *= sx; K[1,2] *= sy

                    try:
                        pts_world = unproject_depth_numpy(d_np, K, extri, mask=mask_valid)
                        if pts_world.shape[0] > 0:
                            pred_world_list.append(pts_world.astype(np.float64))
                    except Exception as e:
                        print(f"   [WARN] failed to unproject predicted depth for frame {frame_idx}: {e}")
                        continue

        if len(pred_world_list) == 0:
            print("  no predicted points for sequence -> skip")
            continue

        pred_world_all = np.concatenate(pred_world_list, axis=0)
        print(f" PRED points built: {pred_world_all.shape[0]} pts")
        stats_info("PRED_ALL_before", pred_world_all)
        stats_info("GT_ALL_before", gt_pts)

        # --- Heuristic unit check (mm vs m) ---
        # Compare medians to guess scale mismatch
        pred_median = float(np.median(np.abs(pred_world_all)))
        gt_median = float(np.median(np.abs(gt_pts)))
        if args.verbose:
            print(f"  [UNIT CHECK] median abs pred={pred_median:.4f}, gt={gt_median:.4f}")

        # If GT is huge (>> pred), consider GT in mm and convert to meters
        # (This is a heuristic; we print action)
        if gt_median > 100.0 and pred_median < 10.0 and gt_median / (pred_median + 1e-9) > 50.0:
            if args.verbose:
                print("  [UNIT FIX] converting GT points from mm -> meters (divide by 1000).")
            gt_pts = gt_pts / 1000.0
            stats_info("GT_after_unit_fix", gt_pts)
            gt_median = float(np.median(np.abs(gt_pts)))

        # If pred seems in mm (very large) and GT small, convert pred
        if pred_median > 100.0 and gt_median < 10.0 and pred_median / (gt_median + 1e-9) > 50.0:
            if args.verbose:
                print("  [UNIT FIX] converting PRED points from mm -> meters (divide by 1000).")
            pred_world_all = pred_world_all / 1000.0
            stats_info("PRED_after_unit_fix", pred_world_all)
            pred_median = float(np.median(np.abs(pred_world_all)))

        # --- Heuristic: detect if predicted points are in camera-frame and require cam->world transform ---
        converted_cam2world = False
        try:
            # get first available extrinsic for testing
            ext0 = get_extrinsic_for_frame(extrinsic_all, 0) if extrinsic_all is not None else None
            if ext0 is None:
                # try to fetch from annotation first entry
                if 'extri' in entries[0]:
                    ext0 = np.array(entries[0]['extri'])
            if ext0 is not None:
                # sample subsets (random sample for speed)
                sample_pred = sample_pts(pred_world_all, n=5000)
                sample_gt = sample_pts(gt_pts, n=5000)
                # compute centroids
                centroid_pred = sample_pred.mean(axis=0)
                centroid_gt = sample_gt.mean(axis=0)
                # transform sample_pred as if it's camera-frame -> world-frame using ext0
                sample_pred_cam2world = cam_to_world(sample_pred, ext0)
                centroid_pred_c2w = sample_pred_cam2world.mean(axis=0)
                dist_before = np.linalg.norm(centroid_pred - centroid_gt)
                dist_after = np.linalg.norm(centroid_pred_c2w - centroid_gt)
                print(f"  [FRAME POSE CHECK] centroid dist before={dist_before:.6f}, after(cam2world)={dist_after:.6f}")
                # if converting decreases centroid distance significantly, perform per-frame cam->world transform
                if dist_after < dist_before * 0.9:
                    print("  [COORD FIX] transforming predicted points from camera-frame -> world-frame using per-frame extrinsics.")
                    new_pred_list = []
                    for fi, pts in enumerate(pred_world_list):
                        extf = get_extrinsic_for_frame(extrinsic_all, fi)
                        if extf is None:
                            # fallback to annotation's extrinsic for the frame
                            entry = entries[fi]
                            if 'extri' in entry:
                                extf = np.array(entry['extri'])
                            elif 'R' in entry and 'T' in entry:
                                Rpt = np.array(entry['R']); Tpt = np.array(entry['T'])
                                Tpt[:2] *= -1; Rpt[:, :2] *= -1; Rpt = Rpt.transpose(1,0)
                                extf = np.hstack((Rpt, Tpt[:,None]))
                            else:
                                extf = None
                        if extf is not None:
                            try:
                                pts_w = cam_to_world(pts, extf)
                                new_pred_list.append(pts_w)
                            except Exception:
                                new_pred_list.append(pts)
                        else:
                            new_pred_list.append(pts)
                    pred_world_all = np.concatenate(new_pred_list, axis=0)
                    stats_info("PRED_ALL_after_cam2world", pred_world_all)
                    converted_cam2world = True
        except Exception as e:
            print("  [WARN] frame pose check failed (ignored):", e)

        # cap pred points to max_points
        if pred_world_all.shape[0] > args.max_points:
            idxs = np.random.choice(pred_world_all.shape[0], args.max_points, replace=False)
            pred_world_all = pred_world_all[idxs]

        # subsample GT to stable size
        gt_sub = gt_pts
        if gt_sub.shape[0] > args.max_points:
            idxs = np.random.choice(gt_sub.shape[0], args.max_points, replace=False)
            gt_sub = gt_sub[idxs]

        # diagnostics before alignment
        s_pred_before = sample_pts(pred_world_all, n=20000)
        s_gt_before = sample_pts(gt_sub, n=20000)
        acc_b, comp_b, cham_b = chamfer_metrics(s_pred_before[s_pred_before[:,2] > 0], s_gt_before)
        diag = bbox_diag(s_gt_before)
        if args.verbose:
            print(f"[DIAG] BEFORE ALIGN (sample): acc={acc_b:.6f}, comp={comp_b:.6f}, chamfer={cham_b:.6f}, gt_bbox_diag={diag:.6f}")
            if diag > 0:
                print(f"[DIAG] BEFORE ALIGN (normalized): acc={acc_b/diag:.6e}, comp={comp_b/diag:.6e}, chamfer={cham_b/diag:.6e}")

        try:
            R_u, t_u, s_u = umeyama_align(pred_world_all, gt_sub)
            pred_aligned = (s_u * (R_u @ pred_world_all.T).T) + t_u
        except Exception as e:
            print("  [WARN] Umeyama align failed:", e)
            pred_aligned = pred_world_all

        acc, comp, chamfer = chamfer_metrics(pred_aligned, gt_sub)
        results[seq_name] = {'accuracy': acc, 'completeness': comp, 'chamfer': chamfer,
                             'n_pred': pred_world_all.shape[0], 'n_gt': gt_sub.shape[0],
                             'converted_cam2world': converted_cam2world}
        print(f" metrics: acc={acc:.6f}, comp={comp:.6f}, chamfer={chamfer:.6f}, n_pred={pred_world_all.shape[0]}, n_gt={gt_sub.shape[0]}")

    # summary
    if len(results) > 0:
        accs = np.array([v['accuracy'] for v in results.values() if np.isfinite(v['accuracy'])])
        comps = np.array([v['completeness'] for v in results.values() if np.isfinite(v['completeness'])])
        chs = np.array([v['chamfer'] for v in results.values() if np.isfinite(v['chamfer'])])
        print("\nSUMMARY:")
        print(" mean accuracy:", np.mean(accs))
        print(" mean completeness:", np.mean(comps))
        print(" mean chamfer:", np.mean(chs))
    else:
        print("No results computed.")

if __name__ == '__main__':
    main()
