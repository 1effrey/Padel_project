import json, cv2, numpy as np
from core.ball_eval import _build_detector

cfg = json.load(open('config-side1.json'))
det = _build_detector(cfg)
print("device=", det.device, "thr=", det.heatmap_threshold,
      "net=", det.net_w, "x", det.net_h, "in_frames=", det.in_frames)

cap = cv2.VideoCapture(cfg['source'])
print("video opened:", cap.isOpened(), "frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(30):
    ok, f = cap.read()
    if not ok:
        print(i, "READ FAILED"); break
    d = det.detect(f)
    hm = det._infer() if len(det._buffer) >= det.in_frames else None
    hmax = float(np.nanmax(hm)) if hm is not None else -1.0
    hasnan = bool(np.isnan(hm).any()) if hm is not None else False
    print(f"f{i:02d} frame_mean={f.mean():6.1f} found={d.found!s:5} "
          f"reason={getattr(d,'reason',None)!s:8} heatmap_max={hmax:.4f} nan={hasnan}")
cap.release()
