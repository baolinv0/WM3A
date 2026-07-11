#!/usr/bin/env python3
from __future__ import annotations

import argparse
from active_mef.data.builders import build_sequence_pool_manifest, build_hdr_video_manifest


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sequence_pool", help="SICE/Kalantari/other precomputed exposure stack")
    s.add_argument("--input-root", required=True)
    s.add_argument("--gt-root", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--ordinal-step", type=float, default=1.0)
    s.add_argument("--gt-in-subdir", action="store_true")

    h = sub.add_parser("hdr_video", help="HDR-GT frame sequences for exposure simulation")
    h.add_argument("--hdr-root", required=True)
    h.add_argument("--output", required=True)
    h.add_argument("--frame-dt", type=float, default=1.0 / 30.0)
    h.add_argument("--stride", type=int, default=1)

    args = p.parse_args()
    if args.cmd == "sequence_pool":
        n = build_sequence_pool_manifest(
            args.input_root, args.gt_root, args.output, args.ordinal_step, args.gt_in_subdir
        )
    else:
        n = build_hdr_video_manifest(args.hdr_root, args.output, args.frame_dt, args.stride)
    print(f"Wrote {n} records to {args.output}")


if __name__ == "__main__":
    main()
