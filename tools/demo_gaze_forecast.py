#!/usr/bin/env python3
"""
Lightweight demo script for CSTS gaze forecasting.

Given a pre-trained checkpoint and a short video clip (ideally the 5-second
clips produced by data/preprocess.py), the script:
  * extracts the early clip frames + matching audio spectrogram,
  * runs the model to predict future gaze heatmaps,
  * writes normalized coordinates to JSON and optional visualization frames.

Example:
python tools/demo_gaze_forecast.py \
  --video-path /path/to/clip.mp4 \
  --checkpoint /path/to/checkpoint_epoch_00005.pyth \
  --config configs/Ego4D/CSTS_Ego4D_Gaze_Forecast.yaml \
  --output-dir demo_outputs
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import librosa
import numpy as np
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
from slowfast.config.defaults import assert_and_infer_cfg, get_cfg
from slowfast.datasets import decoder, video_container
from slowfast.datasets import utils as data_utils
from slowfast.models import build_model
from slowfast.utils.utils import frame_softmax


logger = logging.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo: run CSTS gaze forecasting on a single clip.")
    parser.add_argument("--video-path", required=True, help="Path to an input video clip (mp4).")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained CSTS checkpoint (.pyth).")
    parser.add_argument(
        "--config",
        default="configs/Ego4D/CSTS_Ego4D_Gaze_Forecast.yaml",
        help="Config yaml used for inference.",
    )
    parser.add_argument(
        "--output-dir",
        default="demo_outputs",
        help="Directory to store predictions and visualization frames.",
    )
    parser.add_argument(
        "--frames-length",
        type=int,
        default=86,
        help="How many early frames to sample for conditioning (matches training default).",
    )
    parser.add_argument(
        "--softmax-temp",
        type=float,
        default=2.0,
        help="Temperature for frame_softmax, keep at 2.0 to mirror test loop.",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip writing per-frame PNG overlays.",
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU inference even if CUDA is available.",
    )
    return parser.parse_args()


def compute_spectrogram(
    video_path: str,
    clip_frames: torch.Tensor,
    clip_frame_limit: int,
    video_frame_length: int,
) -> torch.Tensor:
    """
    Build an audio spectrogram aligned to the sampled frames.
    Mirrors data/preprocess.py: 24 kHz mono, 10 ms window, 5 ms stride, log-power.
    """
    samples, sample_rate = librosa.load(video_path, sr=24000, mono=True)
    nperseg = int(round(10 * sample_rate / 1e3))
    noverlap = int(round(5 * sample_rate / 1e3))
    spec = librosa.stft(
        samples,
        n_fft=511,
        window="hann",
        hop_length=noverlap,
        win_length=nperseg,
        pad_mode="constant",
    )
    spec = np.log(np.real(spec * np.conj(spec)) + 1e-6)

    # Trim to avoid using information from later in the clip.
    spec = spec[:, : max(int(spec.shape[1] * clip_frame_limit / max(video_frame_length, 1)), 1)]
    if spec.shape[1] < 256:
        spec = np.pad(spec, ((0, 0), (0, 256 - spec.shape[1])), mode="edge")

    frame_indices = clip_frames.clone().float()
    center_pad = min(128, (spec.shape[1] - 1) // 2)
    frame_indices = torch.clamp(
        torch.round((frame_indices / max(clip_frame_limit, 1)) * spec.shape[1]).long(),
        center_pad,
        spec.shape[1] - 1 - center_pad,
    )

    windows = []
    for idx in frame_indices:
        window = spec[:, idx - center_pad : idx + center_pad]
        if window.shape[1] < 256:
            window = np.pad(window, ((0, 0), (0, 256 - window.shape[1])), mode="edge")
        windows.append(window)

    audio_frames = np.stack(windows, axis=0)[np.newaxis, ...]  # C x T x H x W
    return torch.as_tensor(audio_frames).float()


def decode_target_frames(
    video_path: str,
    target_indices: np.ndarray,
    cfg,
) -> torch.Tensor:
    """
    Fetch frames that correspond to the model's prediction horizon so we can render overlays.
    """
    video_cont = video_container.get_video_container(
        video_path, cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE, cfg.DATA.DECODING_BACKEND
    )
    timebase = video_cont.streams.video[0].duration / video_cont.streams.video[0].frames
    video_start_pts = int(target_indices[0] * timebase)
    video_end_pts = int(target_indices[-1] * timebase)

    frames, _ = decoder.pyav_decode_stream(
        container=video_cont,
        start_pts=video_start_pts,
        end_pts=video_end_pts,
        stream=video_cont.streams.video[0],
        stream_name={"video": 0},
    )
    video_cont.close()
    stacked = torch.as_tensor(np.stack([f.to_rgb().to_ndarray() for f in frames]))
    sampled = decoder.temporal_sampling(stacked, 0, stacked.size(0) - 1, cfg.DATA.NUM_FRAMES)
    return sampled.permute(3, 0, 1, 2).float() / 255.0


def prepare_inputs(
    cfg,
    video_path: str,
    clip_frame_limit: int,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
    """
    Decode visual/audio inputs and slice the future frames to visualize predictions.
    """
    clip_frame_limit = max(clip_frame_limit, cfg.DATA.NUM_FRAMES)
    video_cont = video_container.get_video_container(
        video_path, cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE, cfg.DATA.DECODING_BACKEND
    )
    ori_frame_length = video_cont.streams.video[0].frames
    fps = float(video_cont.streams.video[0].average_rate)
    clip_frame_limit = min(clip_frame_limit, ori_frame_length)

    frames, frames_idx = decoder.decode(
        container=video_cont,
        sampling_rate=cfg.DATA.SAMPLING_RATE,
        num_frames=cfg.DATA.NUM_FRAMES,
        clip_idx=1,  # deterministic center clip for demo
        num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS,
        video_meta={},
        target_fps=cfg.DATA.TARGET_FPS,
        backend=cfg.DATA.DECODING_BACKEND,
        max_spatial_scale=cfg.DATA.TEST_CROP_SIZE,
        use_offset=cfg.DATA.USE_OFFSET_SAMPLING,
        get_frame_idx=True,
        frames_length_limit=clip_frame_limit,
    )
    if frames is None or frames_idx is None:
        raise RuntimeError(f"Failed to decode video {video_path}")

    audio_frames = compute_spectrogram(video_path, frames_idx, clip_frame_limit, ori_frame_length)

    # Future indices for the forecast horizon.
    if clip_frame_limit >= ori_frame_length:
        target_indices = np.linspace(0, ori_frame_length - 1, cfg.DATA.NUM_FRAMES).astype("int64")
    else:
        target_indices = np.arange(clip_frame_limit, ori_frame_length)
        target_indices = np.linspace(target_indices[0], target_indices[-1], cfg.DATA.NUM_FRAMES).astype(
            "int64"
        )

    target_frames = decode_target_frames(video_path, target_indices, cfg)

    frames = data_utils.tensor_normalize(frames, cfg.DATA.MEAN, cfg.DATA.STD)
    frames = frames.permute(3, 0, 1, 2)  # C x T x H x W

    # Apply the same spatial crop/flip to targets for clean overlays.
    combined = torch.cat([frames, target_frames], dim=1)
    combined, _ = data_utils.spatial_sampling(
        combined,
        spatial_idx=1,
        min_scale=cfg.DATA.TEST_CROP_SIZE,
        max_scale=cfg.DATA.TEST_CROP_SIZE,
        crop_size=cfg.DATA.TEST_CROP_SIZE,
        random_horizontal_flip=cfg.DATA.RANDOM_FLIP,
        inverse_uniform_sampling=cfg.DATA.INV_UNIFORM_SAMPLE,
    )
    frames, target_frames = (
        combined[:, : frames.size(1), :, :],
        combined[:, frames.size(1) :, :, :],
    )

    inputs = data_utils.pack_pathway_output(cfg, frames)
    inputs = [x.unsqueeze(0).float() for x in inputs]  # add batch dim
    audio_frames = audio_frames.unsqueeze(0)  # B x 1 x T x H x W

    meta = {
        "input_frame_indices": frames_idx.tolist(),
        "prediction_frame_indices": target_indices.tolist(),
        "fps": fps,
    }
    return inputs, audio_frames, target_frames, meta


def heatmaps_to_points(heatmaps: torch.Tensor) -> List[Dict[str, float]]:
    """
    Convert heatmaps (B x C x T x H x W) to normalized xy coordinates.
    """
    heatmaps_np = heatmaps.squeeze(1).cpu().numpy()
    coords: List[Dict[str, float]] = []
    for t in range(heatmaps_np.shape[1]):
        hm = heatmaps_np[:, t, :, :][0]
        y, x = np.unravel_index(hm.argmax(), hm.shape)
        coords.append(
            {
                "x": float((x + 0.5) / hm.shape[1]),
                "y": float((y + 0.5) / hm.shape[0]),
                "max_prob": float(hm.max()),
            }
        )
    return coords


def save_predictions(
    output_dir: Path,
    base_name: str,
    points: List[Dict[str, float]],
    meta: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_points = []
    for idx, point in enumerate(points):
        frame_idx = meta["prediction_frame_indices"][idx]
        merged_points.append(
            {
                "frame_index": int(frame_idx),
                "time_sec": float(frame_idx / max(meta["fps"], 1e-6)),
                **point,
            }
        )
    payload = {
        "video": base_name,
        "fps": meta["fps"],
        "prediction_frame_indices": meta["prediction_frame_indices"],
        "input_frame_indices": meta["input_frame_indices"],
        "points": merged_points,
    }
    with open(output_dir / f"{base_name}_predictions.json", "w") as f:
        json.dump(payload, f, indent=2)


def save_visualizations(
    output_dir: Path,
    base_name: str,
    target_frames: torch.Tensor,
    points: List[Dict[str, float]],
) -> None:
    """
    Render predicted points over the future frames.
    """
    vis_dir = output_dir / f"{base_name}_frames"
    vis_dir.mkdir(parents=True, exist_ok=True)

    frames_np = (target_frames.clamp(0.0, 1.0).permute(1, 2, 3, 0).cpu().numpy() * 255.0).astype(
        np.uint8
    )
    for idx, frame in enumerate(frames_np):
        px = int(points[idx]["x"] * frame.shape[1])
        py = int(points[idx]["y"] * frame.shape[0])
        frame = cv2.circle(frame.copy(), (px, py), radius=6, color=(0, 0, 255), thickness=2)
        cv2.imwrite(str(vis_dir / f"frame_{idx:03d}.png"), frame[:, :, ::-1])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    logging.setup_logging(args.output_dir)
    logger.info(f"Running demo on {device}")

    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
    cfg.TEST.ENABLE = True
    cfg.TRAIN.ENABLE = False
    cfg.TEST.BATCH_SIZE = 1
    cfg.NUM_GPUS = 1 if device.type == "cuda" else 0
    cfg = assert_and_infer_cfg(cfg)

    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model = model.to(device)
    model.eval()

    inputs, audio_frames, target_frames, meta = prepare_inputs(cfg, args.video_path, args.frames_length)
    inputs = [x.to(device) for x in inputs]
    audio_frames = audio_frames.to(device)

    with torch.no_grad():
        preds = model(inputs, audio_frames)
        preds = frame_softmax(preds, temperature=args.softmax_temp)

    points = heatmaps_to_points(preds)
    base_name = Path(args.video_path).stem
    output_dir = Path(args.output_dir)

    save_predictions(output_dir, base_name, points, meta)
    np.save(output_dir / f"{base_name}_heatmaps.npy", preds.cpu().numpy())
    if not args.no_vis:
        save_visualizations(output_dir, base_name, target_frames, points)

    logger.info(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
