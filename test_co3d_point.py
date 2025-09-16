#!/usr/bin/env python3
"""
test_co3d_point.py - Use model outputs (prefer world_points) for CO3Dv2 point-map eval.

e.g.,)
python .\test_co3d_point.py 
--co3d_dir F:\Datasets\Blind_images_v2  
--co3d_anno_dir . 
--model_path .\model_tracker_fixed_e20.pt 
--category backpack 
--sequence 159_17477_32932 
--gt_ply F:\Datasets\Blind_images_v2\backpack\159_17477_32932\pointcloud.ply 

--
Usage: same CLI options as before. Key new behavior:
 - If preds contains 'world_points' and 'world_points_conf', use them directly (fast & robust).
 - Otherwise fallback to depth -> unproject with robust shape handling.
 - umeyama_align: if src and dst have different sizes, pair src->dst by nearest neighbor (KDTree) before alignment.

 모델 출력에서 이미 존재하는 world_points / world_points_conf를 우선 사용합니다. (이번 덤프에서는 이 값이 있어서, 별도 unproject 실패 문제를 우회할 수 있습니다.)

world_points가 없을 때만 depth + pose → unproject_depth_map_to_point_map을 시도합니다. (이전엔 unproject 시도에서 형태 불일치로 계속 실패했음.)

다양한 텐서/넘파이 차원(배치, 시퀀스, 채널 등)을 안전하게 추출하는 헬퍼 구현 (get_frame_from_tensor, tensor_frame_to_map 등).

디버그 옵션 --dump_preds_shapes 유지 (이미 출력한 정보를 얻을 수 있음).

기타: conf 마스크 적용, 최대 포인트 수 절삭, Umeyama 정렬 및 Chamfer(Accuracy/Completeness) 계산 포함.

해결 방법: 대응(correspondences)이 명시적으로 주어지지 않으면, 가장 실용적인 방법은 최근접 이웃(nearest neighbor) 을 이용해 
한 집합의 각 점에 대해 다른 집합의 최근접 점을 찾아 짝을 만든 뒤 Umeyama를 수행하는 것입니다. 
따라서 umeyama_align을 수정하여 점 개수가 다르면 KDTree로 짝을 만든 뒤 정렬하도록 했습니다.

#!/usr/bin/env python3
"""

import argparse
import gzip
import json
import os
import os.path as osp
import random
import numpy as np
import torch
from scipy.spatial import cKDTree

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

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
    p.add_argument('--point_conf_thresh', type=float, default=0.2)
    p.add_argument('--use_pointmap_branch', action='store_true')
    p.add_argument('--max_points', type=int, default=500000)
    p.add_argument('--gt_ply', type=str, default=None)
    p.add_argument('--gt_dir', type=str, default=None)
    p.add_argument('--dump_preds_shapes', action='store_true')
    return p.parse_args()

def load_annotation(anno_file):
    with gzip.open(anno_file, 'r') as f:
        return json.loads(f.read())

def umeyama_align(src, dst, with_scaling=True):
    """
    Umeyama alignment with NN-pairing fallback when src and dst sizes differ.
    - src: (Ns,3) predicted points (world coords)
    - dst: (Nd,3) gt points
    Returns: R, t, s such that aligned = s * R @ src.T + t
    """
    src = np.asarray(src).astype(np.float64)
    dst = np.asarray(dst).astype(np.float64)
    assert src.ndim == 2 and dst.ndim == 2 and src.shape[1] == 3 and dst.shape[1] == 3

    # if either empty -> error
    if src.shape[0] == 0 or dst.shape[0] == 0:
        raise ValueError("Empty source or destination in umeyama_align")

    # If sizes differ, pair each src point with nearest dst point
    if src.shape[0] != dst.shape[0]:
        tree = cKDTree(dst)
        dists, idx = tree.query(src, k=1)
        dst_paired = dst[idx]      # same length as src
        src_paired = src
    else:
        src_paired = src
        dst_paired = dst

    N = src_paired.shape[0]
    mu_src = src_paired.mean(axis=0)
    mu_dst = dst_paired.mean(axis=0)
    src_c = src_paired - mu_src
    dst_c = dst_paired - mu_dst
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

def load_ply_xyz(ply_path):
    # robust loader (tries plyfile then manual parse) - same as before
    try:
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        if 'vertex' in plydata.elements:
            v = plydata['vertex'].data
            if all(k in v.dtype.names for k in ('x','y','z')):
                pts = np.vstack([v['x'], v['y'], v['z']]).T
                return pts.astype(np.float64)
            else:
                names = v.dtype.names
                if len(names) >= 3:
                    pts = np.vstack([v[names[0]], v[names[1]], v[names[2]]]).T
                    return pts.astype(np.float64)
        raise RuntimeError("plyfile read but no vertex/x,y,z found")
    except Exception:
        pass
    with open(ply_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading PLY header")
            try:
                sline = line.decode('ascii').strip()
            except Exception:
                sline = line.decode('latin1').strip()
            header_lines.append(sline)
            if sline.lower() == 'end_header':
                break
        format_line = None
        vertex_count = None
        in_vertex = False
        vertex_props = []
        for ln in header_lines:
            parts = ln.split()
            if len(parts) == 0:
                continue
            if parts[0] == 'format':
                format_line = parts[1]
            elif parts[0] == 'element' and parts[1] == 'vertex':
                vertex_count = int(parts[2]); in_vertex = True
            elif parts[0] == 'element':
                in_vertex = False
            elif parts[0] == 'property' and in_vertex:
                if len(parts) == 3:
                    ptype, pname = parts[1], parts[2]
                    vertex_props.append((ptype, pname))
        if vertex_count is None or len(vertex_props) == 0:
            raise RuntimeError("No vertex element/properties found in PLY header")
        prop_names = [pn for _, pn in vertex_props]
        try:
            ix = prop_names.index('x'); iy = prop_names.index('y'); iz = prop_names.index('z')
        except ValueError:
            ix, iy, iz = 0,1,2
        type_map = {
            'char': ('b',1), 'uchar':('B',1),'int8':('b',1),'uint8':('B',1),
            'short':('h',2),'ushort':('H',2),'int16':('h',2),'uint16':('H',2),
            'int':('i',4),'int32':('i',4),'uint':('I',4),'uint32':('I',4),
            'float':('f',4),'float32':('f',4),'double':('d',8),'float64':('d',8)
        }
        fmt_chars = []; byte_len = 0
        for ptype, pname in vertex_props:
            key = ptype.lower()
            if key not in type_map:
                raise RuntimeError(f"Unsupported PLY property type: {ptype}")
            ch, sz = type_map[key]
            fmt_chars.append(ch); byte_len += sz
        if format_line is None:
            raise RuntimeError("PLY format not specified")
        if 'binary_little_endian' in format_line: endian = '<'
        elif 'binary_big_endian' in format_line: endian = '>'
        elif 'ascii' in format_line: endian = 'ascii'
        else: raise RuntimeError("Unknown PLY format: " + format_line)
        if endian == 'ascii':
            pts = []
            for i in range(vertex_count):
                line = f.readline().decode('ascii').strip()
                parts = line.split()
                if len(parts) < max(ix,iy,iz)+1: continue
                x = float(parts[ix]); y = float(parts[iy]); z = float(parts[iz])
                pts.append([x,y,z])
            return np.array(pts, dtype=np.float64)
        data = f.read(vertex_count * byte_len)
        if len(data) < vertex_count * byte_len:
            raise RuntimeError("PLY binary data truncated")
        struct_fmt = endian + ''.join(fmt_chars)
        import struct as _struct
        unpack = _struct.unpack_from
        pts = []; offset = 0
        for i in range(vertex_count):
            vals = unpack(struct_fmt, data, offset)
            offset += byte_len
            x = float(vals[ix]); y = float(vals[iy]); z = float(vals[iz])
            pts.append([x,y,z])
        return np.array(pts, dtype=np.float64)

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

def tensor_frame_to_map(tensor, frame_idx, batch_size):
    """robust extraction for common tensor / ndarray shapes (handles 5D produced by model)."""
    if tensor is None:
        return None
    arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if arr.ndim == 5:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            out = arr[0, frame_idx]
            if out.shape[-1] == 1:
                out = np.squeeze(out, axis=-1)
            return out
        if arr.shape[0] == 1:
            a2 = arr.squeeze(0)
            if a2.shape[0] > frame_idx:
                out = a2[frame_idx]
                if out.shape[-1] == 1:
                    out = np.squeeze(out, axis=-1)
                return out
            return np.squeeze(a2)
    if arr.ndim == 4:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            out = arr[0, frame_idx]
            if out.ndim == 3 and out.shape[-1] == 1:
                out = np.squeeze(out, axis=-1)
            return out
        if arr.shape[0] > 1 and arr.shape[0] == frame_idx + 1:
            return arr[frame_idx]
        return np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[0] == frame_idx + 1 or arr.shape[0] == frame_idx:
            return arr[frame_idx]
        if arr.shape[0] == 1:
            return np.squeeze(arr, axis=0)
        return arr
    if arr.ndim == 2:
        return arr
    return np.squeeze(arr)

def get_frame_from_tensor(tensor, idx, batch_n, as_torch=True, device=None, dtype=None):
    if tensor is None:
        return None
    try:
        if isinstance(tensor, torch.Tensor):
            t = tensor
            nd = t.dim()
            if nd >= 3 and t.shape[0] == 1 and t.shape[1] > 1:
                sel = t[0, idx]
                if as_torch and device is not None:
                    sel = sel.to(device=device, dtype=dtype)
                return sel
            if nd >= 3 and t.shape[0] == 1 and t.shape[1] == batch_n:
                sel = t[0, idx]
                if as_torch and device is not None:
                    sel = sel.to(device=device, dtype=dtype)
                return sel
            if nd >= 3 and t.shape[0] == batch_n:
                sel = t[idx]
                if as_torch and device is not None:
                    sel = sel.to(device=device, dtype=dtype)
                return sel
            if nd >= 2 and t.shape[0] == 1:
                sel = t.squeeze(0)
                if as_torch and device is not None:
                    sel = sel.to(device=device, dtype=dtype)
                if sel.dim() >= 1 and sel.shape[0] > 1 and sel.shape[0] == batch_n:
                    return sel[idx]
                return sel
            return t
        else:
            arr = np.asarray(tensor)
            m = tensor_frame_to_map(arr, idx, batch_n)
            if m is None:
                return None
            if as_torch:
                t = torch.from_numpy(m)
                if device is not None:
                    t = t.to(device=device)
                if dtype is not None:
                    t = t.to(dtype=dtype)
                return t
            return m
    except Exception:
        return None

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = args.device
    if device.startswith('cuda'):
        try:
            cc = torch.cuda.get_device_capability()[0]
        except Exception:
            cc = 8
        dtype = torch.bfloat16 if cc >= 8 else torch.float16
    else:
        dtype = torch.float32

    print("Loading model:", args.model_path)
    if args.model_path.startswith("facebook/") or (('/' in args.model_path or '\\' in args.model_path) and not osp.exists(args.model_path)):
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
            raise ValueError(f"Sequence {args.sequence} not found")
        seq_names = [args.sequence]

    if args.gt_ply is None and args.gt_dir is None:
        print("[WARN] No GT provided"); return

    results = {}

    for seq_name in seq_names:
        print("Processing", seq_name)
        seq_data = annotation[seq_name]
        if len(seq_data) < args.min_num_images:
            print(f" skip (only {len(seq_data)} images < min {args.min_num_images})"); continue

        if len(seq_data) <= args.num_frames:
            chosen = list(range(len(seq_data)))
        else:
            chosen = sorted(np.random.choice(len(seq_data), args.num_frames, replace=False).tolist())
        print(" chosen indices:", chosen)

        if args.gt_ply:
            gt_pts = load_ply_xyz(args.gt_ply)
        else:
            candidate = osp.join(args.gt_dir, f"{seq_name}.ply")
            if osp.exists(candidate):
                gt_pts = load_ply_xyz(candidate)
            else:
                files = [f for f in os.listdir(args.gt_dir) if f.lower().endswith('.ply')]
                if len(files) == 1:
                    gt_pts = load_ply_xyz(osp.join(args.gt_dir, files[0]))
                else:
                    raise FileNotFoundError("GT not found")

        image_paths = []
        entries = []
        for idx in chosen:
            entry = seq_data[idx]
            img_path = osp.normpath(osp.join(args.co3d_dir, entry['filepath']))
            image_paths.append(img_path); entries.append(entry)
        missing = [p for p in image_paths if not osp.exists(p)]
        if missing:
            print("  [WARN] missing images:", missing[:5])
        valid_pairs = [(p,e) for p,e in zip(image_paths, entries) if osp.exists(p)]
        if len(valid_pairs) == 0:
            print("  no valid images -> skip"); continue
        image_paths, entries = zip(*valid_pairs)

        imgs_tensor = load_and_preprocess_images(list(image_paths)).to(device)

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
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: torch{tuple(v.shape)} {v.dtype}")
                    else:
                        try:
                            a = np.asarray(v)
                            print(f"  {k}: np{tuple(a.shape)} {a.dtype}")
                        except Exception:
                            print(f"  {k}: type {type(v)}")

        extrinsic_all = None; intrinsic_all = None
        if isinstance(preds, dict) and 'pose_enc' in preds:
            try:
                extrinsic_all, intrinsic_all = pose_encoding_to_extri_intri(preds['pose_enc'], imgs_tensor.shape[-2:])
            except Exception as e:
                print("  [WARN] pose_encoding_to_extri_intri failed:", e)

        if extrinsic_all is None and isinstance(preds, dict):
            for k in ['extrinsic','extri','cam_extri','camera']:
                if k in preds:
                    extrinsic_all = preds[k]; break

        depth_tensor = preds.get('depth') if isinstance(preds, dict) and 'depth' in preds else None
        depth_conf_tensor = preds.get('depth_conf') if isinstance(preds, dict) and 'depth_conf' in preds else None
        point_map_tensor = preds.get('point_map') if isinstance(preds, dict) and 'point_map' in preds else None
        point_conf_tensor = preds.get('point_conf') if isinstance(preds, dict) and 'point_conf' in preds else None
        world_points_tensor = preds.get('world_points') if isinstance(preds, dict) and 'world_points' in preds else None
        world_points_conf = preds.get('world_points_conf') if isinstance(preds, dict) and 'world_points_conf' in preds else None

        pred_world_list = []

        for i, entry in enumerate(entries):
            frame_idx = int(chosen[i])

            used_world_branch = False
            if world_points_tensor is not None:
                try:
                    wp = tensor_frame_to_map(world_points_tensor, i, imgs_tensor.shape[0])
                    wc = tensor_frame_to_map(world_points_conf, i, imgs_tensor.shape[0]) if world_points_conf is not None else None
                    if wp is not None:
                        wp = np.asarray(wp)
                        if wp.ndim == 3 and wp.shape[0] == 3:
                            pts = wp.reshape(3, -1).T
                        elif wp.ndim == 3 and wp.shape[2] == 3:
                            pts = wp.reshape(-1, 3)
                        else:
                            pts = np.reshape(wp, (-1, 3))
                        if wc is not None:
                            wc = np.asarray(wc)
                            if wc.ndim == 3 and wc.shape[-1] == 1:
                                wc = wc[:,:,0]
                            flat_conf = wc.reshape(-1)
                            if flat_conf.shape[0] != pts.shape[0]:
                                minlen = min(flat_conf.shape[0], pts.shape[0])
                                pts = pts[:minlen]; flat_conf = flat_conf[:minlen]
                            pts = pts[flat_conf >= args.point_conf_thresh]
                        if pts.shape[0] > 0:
                            pred_world_list.append(pts.astype(np.float64))
                            used_world_branch = True
                except Exception as e:
                    print(f"   [WARN] world_points branch failed for frame {frame_idx}: {e}")

            if used_world_branch:
                continue

            per_extri = get_frame_from_tensor(extrinsic_all, i, imgs_tensor.shape[0], as_torch=False)
            extri = None
            if per_extri is not None:
                try:
                    extri = per_extri.detach().cpu().numpy() if isinstance(per_extri, torch.Tensor) else np.asarray(per_extri)
                except Exception:
                    extri = None
            if extri is None:
                if 'extri' in entry:
                    extri = np.array(entry['extri'])
                elif 'R' in entry and 'T' in entry:
                    Rpt = np.array(entry['R']); Tpt = np.array(entry['T'])
                    Tpt[:2] *= -1; Rpt[:, :2] *= -1; Rpt = Rpt.transpose(1,0)
                    extri = np.hstack((Rpt, Tpt[:,None]))
                else:
                    print(f"  [WARN] no extrinsic for frame {frame_idx} -> skip"); continue

            per_intr = get_frame_from_tensor(intrinsic_all, i, imgs_tensor.shape[0], as_torch=False)
            intr = None
            if per_intr is not None:
                try:
                    intr = per_intr.detach().cpu().numpy() if isinstance(per_intr, torch.Tensor) else np.asarray(per_intr)
                except Exception:
                    intr = None
            if intr is None and 'focal_length' in entry and 'principal_point' in entry:
                intr = (entry['focal_length'], entry['principal_point'])

            d_np = None; cd_np = None
            try:
                d_np = tensor_frame_to_map(depth_tensor, i, imgs_tensor.shape[0])
                if d_np is not None:
                    d_np = np.asarray(d_np)
                    if d_np.ndim == 3 and d_np.shape[-1] == 1:
                        d_np = d_np[:,:,0]
            except Exception:
                d_np = None
            try:
                cd_np = tensor_frame_to_map(depth_conf_tensor, i, imgs_tensor.shape[0]) if depth_conf_tensor is not None else None
                if cd_np is not None:
                    cd_np = np.asarray(cd_np)
                    if cd_np.ndim == 3 and cd_np.shape[-1] == 1:
                        cd_np = cd_np[:,:,0]
            except Exception:
                cd_np = None

            cam_pts = np.zeros((0,3), dtype=np.float64)

            if d_np is None:
                print(f"   [WARN] cannot extract depth for frame {frame_idx}")
            else:
                d_np = np.squeeze(d_np).astype(np.float64)
                if cd_np is None:
                    cd_np = np.ones_like(d_np)
                else:
                    cd_np = np.squeeze(cd_np)
                    if cd_np.shape != d_np.shape:
                        if cd_np.ndim == 3 and cd_np.shape[2] == 1 and cd_np.shape[0:2] == d_np.shape:
                            cd_np = cd_np[:,:,0]
                        elif cd_np.ndim == 3 and cd_np.shape[0] == 1 and cd_np.shape[1:] == d_np.shape:
                            cd_np = cd_np[0]
                        else:
                            cd_np = np.ones_like(d_np)
                mask_valid = (cd_np >= args.depth_conf_thresh) & (d_np > 0) & (~np.isnan(d_np))
                if mask_valid.sum() == 0:
                    print(f"   [WARN] no valid depth pixels for frame {frame_idx}")
                else:
                    depth_t = torch.from_numpy(d_np).to(device=device, dtype=imgs_tensor.dtype)
                    e_np = extri
                    if e_np.shape == (4,4):
                        e_34 = e_np[:3,:4]
                    elif e_np.shape == (3,4):
                        e_34 = e_np
                    elif e_np.shape == (1,3,4) or e_np.shape == (1,4,4):
                        e_ = np.squeeze(e_np, axis=0)
                        if e_.shape == (4,4): e_34 = e_[:3,:4]
                        else: e_34 = e_
                    else:
                        try:
                            e_34 = np.reshape(e_np, (3,4))
                        except Exception:
                            e_34 = None
                    if e_34 is None:
                        print(f"   [WARN] cannot coerce extrinsic for frame {frame_idx} -> skip")
                    else:
                        extr_t = torch.from_numpy(e_34).to(device=device, dtype=imgs_tensor.dtype)
                        K_t = None
                        if isinstance(intr, (list,tuple)) and len(intr)==2:
                            focal, pp = intr
                            fx = float(focal[0]) if hasattr(focal, '__len__') else float(focal)
                            fy = float(focal[1]) if hasattr(focal, '__len__') and len(focal)>1 else fx
                            cx = float(pp[0]); cy = float(pp[1])
                            K = np.array([[fx,0.0,cx],[0.0,fy,cy],[0.0,0.0,1.0]], dtype=np.float64)
                            K_t = torch.from_numpy(K).to(device=device, dtype=imgs_tensor.dtype)
                        elif intr is not None and isinstance(intr, np.ndarray) and intr.shape == (3,3):
                            K_t = torch.from_numpy(intr).to(device=device, dtype=imgs_tensor.dtype)
                        try:
                            pm = unproject_depth_map_to_point_map(depth_t, extr_t, K_t)
                            pm_arr = pm.detach().cpu().numpy() if isinstance(pm, torch.Tensor) else np.asarray(pm)
                            if pm_arr.ndim == 3 and pm_arr.shape[0] == 3:
                                cam_all = pm_arr.reshape(3, -1).T
                            elif pm_arr.ndim == 3 and pm_arr.shape[2] == 3:
                                cam_all = pm_arr.reshape(-1, 3)
                            else:
                                cam_all = np.reshape(pm_arr, (-1, 3))
                            flat_mask = mask_valid.reshape(-1)
                            if flat_mask.shape[0] != cam_all.shape[0]:
                                minlen = min(flat_mask.shape[0], cam_all.shape[0])
                                cam_all = cam_all[:minlen]; flat_mask = flat_mask[:minlen]
                            cam_pts = cam_all[flat_mask.astype(bool)]
                        except Exception as e:
                            print(f"   [WARN] unproject failed for frame {frame_idx}: {e}")

            if cam_pts.shape[0] == 0 and args.use_pointmap_branch and point_map_tensor is not None:
                try:
                    pm_arr = tensor_frame_to_map(point_map_tensor, i, imgs_tensor.shape[0])
                    pm_arr = np.asarray(pm_arr)
                    if pm_arr.ndim == 3 and pm_arr.shape[0] == 3:
                        branch_pts = pm_arr.reshape(3, -1).T
                    elif pm_arr.ndim == 3 and pm_arr.shape[2] == 3:
                        branch_pts = pm_arr.reshape(-1, 3)
                    else:
                        branch_pts = np.reshape(pm_arr, (-1,3))
                    if point_conf_tensor is not None:
                        pc_arr = tensor_frame_to_map(point_conf_tensor, i, imgs_tensor.shape[0])
                        pc_arr = np.squeeze(np.asarray(pc_arr)); pc_flat = pc_arr.reshape(-1)
                        if pc_flat.shape[0] != branch_pts.shape[0]:
                            minlen = min(pc_flat.shape[0], branch_pts.shape[0])
                            branch_pts = branch_pts[:minlen]; pc_flat = pc_flat[:minlen]
                        branch_pts = branch_pts[pc_flat >= args.point_conf_thresh]
                    if branch_pts is not None and branch_pts.shape[0] > 0:
                        cam_pts = branch_pts
                except Exception as e:
                    print(f"   [WARN] point_map branch failed for frame {frame_idx}: {e}")

            if cam_pts.shape[0] == 0:
                print(f"   frame {frame_idx}: no cam points -> skip")
                continue

            R = extri[:3, :3]; t = extri[:3, 3]; Rinv = R.T
            pts_world = (Rinv @ (cam_pts.T - t[:, None])).T
            pred_world_list.append(pts_world)

        if len(pred_world_list) == 0:
            print("  no predicted points for sequence -> skip"); continue

        pred_world_all = np.concatenate(pred_world_list, axis=0)
        if pred_world_all.shape[0] > args.max_points:
            idxs = np.random.choice(pred_world_all.shape[0], args.max_points, replace=False)
            pred_world_all = pred_world_all[idxs]

        # Here: pred_world_all and gt_pts can differ in size. umeyama_align now handles pairing.
        R_u, t_u, s_u = umeyama_align(pred_world_all, gt_pts)
        pred_aligned = (s_u * (R_u @ pred_world_all.T).T) + t_u
        acc, comp, chamfer = chamfer_metrics(pred_aligned, gt_pts)
        results[seq_name] = {'accuracy': acc, 'completeness': comp, 'chamfer': chamfer,
                             'n_pred': pred_world_all.shape[0], 'n_gt': gt_pts.shape[0]}
        print(f" metrics: acc={acc:.4f}, comp={comp:.4f}, chamfer={chamfer:.4f}")

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
