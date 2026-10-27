import cv2, os, sys, glob

VIDEO_EXTS = (".mov", ".mp4", ".avi", ".mkv", ".m4v", ".3gp", ".wmv")
FRAMES_PER_VIDEO = 60

name = sys.argv[1]                                   # e.g. Henish_Prajapati
frames_per_video = int(sys.argv[2]) if len(sys.argv) > 2 else FRAMES_PER_VIDEO

src = os.path.join("videos", name)
out = os.path.join("captures", name)
os.makedirs(out, exist_ok=True)

videos = sorted(p for p in glob.glob(os.path.join(src, "*"))
                if p.lower().endswith(VIDEO_EXTS))
if not videos:
    sys.exit(f"No video files found in {src}")

existing = [f for f in os.listdir(out) if f.startswith("sample_")]
n = max([int(os.path.splitext(f)[0].split("_")[-1]) for f in existing], default=0)

for video in videos:
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or total <= 0:
        print(f"!! Could not read {os.path.basename(video)} - see the note about converting it.")
        continue

    # pick frames evenly spaced across the whole video
    count = min(frames_per_video, total)
    targets = {int(i * total / count) for i in range(count)}

    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in targets:
            h, w = frame.shape[:2]
            if max(h, w) > 1280:                      # shrink large phone frames
                s = 1280 / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)))
            n += 1
            saved += 1
            cv2.imwrite(os.path.join(out, f"sample_{n}.jpg"), frame)
        i += 1
    cap.release()
    print(f"{os.path.basename(video)}: {total} frames in video, saved {saved}")

print(f"Done. Folder now has photos up to sample_{n}.jpg")