"""
Group 19 - Real-time Face Recognition with ArcFace
==================================================
Aditya Gajjar (KU2407U392), Henish Prajapati (KU2407U404),
Krish Dhairav (KU2407U416), Shah Chaitya (KU2407U503)

Pipeline (matches Section V of our report)
------------------------------------------
1. Detection  : SCRFD face detector -> bounding box + 5 facial landmarks
2. Alignment  : 2D similarity transform of the 5 landmarks onto the standard
                ArcFace template -> 112x112 aligned face
3. Features   : ArcFace ResNet-50 backbone (pretrained by InsightFace with the
                additive angular margin loss) -> 512-d L2-normalised embedding.
                The backbone is FROZEN: training a deep CNN from scratch needs
                millions of images, we only have a few hundred.
4. Classifier : our own ArcFace head (Eq. 3 of the report, s = 64, m = 0.5)
                trained on our four members' embeddings. Its weight vectors W_j
                are the learned "class centres" on the hypersphere.
5. Open set   : cos(theta) = embedding . W_j. If the best cosine is below a
                threshold tuned on the validation set -> "Unidentified".

Usage
-----
  python Group_19_1.py capture --name Aditya_Gajjar   # webcam photos of one member
  python Group_19_1.py split                          # captures/ -> data/train|validate|test
  python Group_19_1.py train                          # train ArcFace head + tune threshold
  python Group_19_1.py test                           # accuracy on the test set
  python Group_19_1.py run                            # live webcam demo

Optional (recommended): photos of people who are NOT in the group, e.g. friends,
  unknown/validate/*.jpg  -> used to tune the "Unidentified" threshold
  unknown/test/*.jpg      -> used only to report accuracy on strangers
Use different strangers in the two folders.

Requirements:  pip install insightface onnxruntime opencv-python torch numpy
(First run downloads the pretrained "buffalo_l" model pack, ~280 MB, into
~/.insightface/models. Do this once with internet before the demo.)
"""

import argparse
import os
import random
import shutil
import sys
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)   # scikit-image notice inside insightface

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CAPTURE_DIR = os.path.join(SCRIPT_DIR, "captures")    # raw webcam photos before split
UNKNOWN_DIR = os.path.join(SCRIPT_DIR, "unknown")     # optional: photos of non-members
MODEL_PATH = os.path.join(SCRIPT_DIR, "Group_19_model.pt")
SPLITS = ("train", "validate", "test")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SEED = 19

ARC_S = 64.0   # feature scale s  (report Eq. 3)
ARC_M = 0.5    # angular margin m (report Eq. 3), in radians (~28.6 degrees)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def sample_number(filename):
    """'sample_12.jpg' -> 12 (used to keep files in capture order)."""
    stem = os.path.splitext(filename)[0]
    digits = stem.split("_")[-1]
    return int(digits) if digits.isdigit() else 0


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
    files.sort(key=lambda f: (sample_number(f), f))
    return [os.path.join(folder, f) for f in files]


def list_members(split_dir):
    if not os.path.isdir(split_dir):
        return []
    return sorted(d for d in os.listdir(split_dir)
                  if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith("."))


def display_name(member):
    return member.replace("_", " ")


# --------------------------------------------------------------------------
# Face engine: detection -> 5-point alignment -> ArcFace embedding
# --------------------------------------------------------------------------
class FaceEngine:
    def __init__(self, det_size=640, model_pack="buffalo_l"):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align

        self.face_align = face_align
        use_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if use_gpu else ["CPUExecutionProvider"])
        app = FaceAnalysis(name=model_pack,
                           allowed_modules=["detection", "recognition"],
                           providers=providers)
        app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))
        self.detector = app.models["detection"]        # SCRFD
        self.recognizer = app.models["recognition"]    # ArcFace ResNet-50 (512-d)
        print(f"[engine] {model_pack} loaded on {'GPU' if use_gpu else 'CPU'}, "
              f"detector input {det_size}x{det_size}")

    def detect(self, img):
        """Return a list of (box[x1,y1,x2,y2], landmarks 5x2, score), largest face first."""
        bboxes, kpss = self.detector.detect(img, max_num=0)
        faces = []
        if bboxes is None or len(bboxes) == 0:
            return faces
        for i in range(bboxes.shape[0]):
            faces.append((bboxes[i, :4], kpss[i], float(bboxes[i, 4])))
        faces.sort(key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]), reverse=True)
        return faces

    def align(self, img, landmarks):
        """2D alignment: map the 5 landmarks onto the ArcFace 112x112 template."""
        return self.face_align.norm_crop(img, landmark=landmarks, image_size=112)

    def embed(self, img, landmarks):
        """512-d L2-normalised ArcFace embedding of one face."""
        aligned = self.align(img, landmarks)
        feat = self.recognizer.get_feat(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-10)

    def largest_face_embedding(self, img):
        faces = self.detect(img)
        if not faces:
            return None
        return self.embed(img, faces[0][1])


# --------------------------------------------------------------------------
# Data augmentation (applied to full photos, before detection)
# --------------------------------------------------------------------------
def augment(img, rng):
    out = img.copy()
    h, w = out.shape[:2]
    # horizontal flip
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    # rotation (+-15 deg) and scale (0.75x - 1.15x): simulates head tilt and distance
    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.75, 1.15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    # brightness / contrast: simulates different lighting
    alpha = rng.uniform(0.6, 1.4)
    beta = rng.uniform(-40, 40)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    # occasional blur (motion / focus) and sensor noise
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.3:
        noise = rng.normal(0, 8, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def embed_folder_set(engine, split_dir, members, n_aug=0, verbose=True):
    """Embed every image of every member in split_dir. Returns X (N,512), y (N,)."""
    rng = np.random.default_rng(SEED)
    X, y = [], []
    for label, member in enumerate(members):
        paths = list_images(os.path.join(split_dir, member))
        kept, skipped = 0, 0
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                skipped += 1
                continue
            versions = [img] + [augment(img, rng) for _ in range(n_aug)]
            for k, v in enumerate(versions):
                e = engine.largest_face_embedding(v)
                if e is None:
                    if k == 0:
                        skipped += 1
                    continue
                X.append(e)
                y.append(label)
                kept += 1
        if verbose:
            aug_txt = f" (incl. {n_aug} augmented copies per photo)" if n_aug else ""
            print(f"   {member:<22} photos: {len(paths):4d}  embeddings: {kept:5d}{aug_txt}"
                  + (f"  [no face found in {skipped}]" if skipped else ""))
    if not X:
        return np.zeros((0, 512), np.float32), np.zeros((0,), np.int64)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


# --------------------------------------------------------------------------
# ArcFace head: additive angular margin loss (report Eq. 3)
# --------------------------------------------------------------------------
def build_arcface_head(in_dim, n_classes, s=ARC_S, m=ARC_M):
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ArcFaceHead(nn.Module):
        """
        logits_j = s * cos(theta_j)          for j != y
        logits_y = s * cos(theta_y + m)      for the true class y
        where cos(theta_j) = normalize(x) . normalize(W_j)
        """

        def __init__(self):
            super().__init__()
            self.s, self.m = s, m
            self.W = nn.Parameter(torch.empty(n_classes, in_dim))
            nn.init.xavier_uniform_(self.W)
            self.cos_m, self.sin_m = math.cos(m), math.sin(m)
            # if theta + m would exceed pi, cos(theta+m) stops being monotonic;
            # fall back to a CosFace-style penalty there (as in the official code)
            self.th = math.cos(math.pi - m)
            self.mm = math.sin(math.pi - m) * m

        def cosine(self, x):
            return F.linear(F.normalize(x), F.normalize(self.W))

        def forward(self, x, labels):
            cos = self.cosine(x).clamp(-1 + 1e-7, 1 - 1e-7)
            sin = torch.sqrt(1.0 - cos ** 2)
            cos_theta_plus_m = cos * self.cos_m - sin * self.sin_m   # cos(theta + m)
            cos_theta_plus_m = torch.where(cos > self.th, cos_theta_plus_m, cos - self.mm)
            one_hot = F.one_hot(labels, n_classes).float()
            logits = self.s * (one_hot * cos_theta_plus_m + (1 - one_hot) * cos)
            return logits

    return ArcFaceHead()


def train_head(Xtr, ytr, Xva, yva, n_classes, epochs=300, lr=0.01, patience=40):
    import torch
    import torch.nn.functional as F

    head = build_arcface_head(Xtr.shape[1], n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    batch = 64
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    history = []

    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(Xtr_t))
        total = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(head(Xtr_t[idx], ytr_t[idx]), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        train_loss = total / len(Xtr_t)

        head.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(head(Xva_t, yva_t), yva_t).item()
            val_acc = (head.cosine(Xva_t).argmax(1) == yva_t).float().mean().item()
        history.append((train_loss, val_loss, val_acc))

        if val_loss < best_loss - 1e-4:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"   epoch {epoch:3d}  train loss {train_loss:7.3f}  "
                  f"val loss {val_loss:7.3f}  val acc {val_acc * 100:6.2f}%")
        if bad_epochs >= patience:
            print(f"   early stop at epoch {epoch} (best val loss {best_loss:.3f})")
            break

    head.load_state_dict(best_state)
    return head, history


def class_centres(head):
    """Normalised ArcFace weight vectors = learned class centres (n_classes x 512)."""
    W = head.W.detach().cpu().numpy()
    return W / np.linalg.norm(W, axis=1, keepdims=True)


MIN_THRESHOLD = 0.30   # safety floor: strangers rarely exceed ~0.25 cosine with this model


def choose_threshold(cos_va, yva, stranger_scores=None):
    """
    Tune the 'Unidentified' threshold on the validation set.
      genuine score  : cosine of a member's photo to their own class centre
      impostor score : (a) best cosine of a member's photo to any OTHER class
                       centre (leave-one-class-out: how they would score if
                       not enrolled), plus (b) best cosine of real strangers
                       from unknown/validate/ if provided.
    (a) alone is optimistic, because the ArcFace loss explicitly pushed the
    other centres away from these exact people, so real strangers matter.

    Rule: put the threshold in the middle of the gap between the highest
    impostor score and the low end (2nd percentile) of the genuine scores,
    never below MIN_THRESHOLD unless genuine scores themselves are that low.
    """
    n = len(yva)
    genuine = cos_va[np.arange(n), yva]
    masked = cos_va.copy()
    masked[np.arange(n), yva] = -np.inf
    impostor = masked.max(1)
    if stranger_scores is not None and len(stranger_scores):
        impostor = np.concatenate([impostor, stranger_scores])

    hi_impostor = float(impostor.max())
    lo_genuine = float(np.percentile(genuine, 2))
    if hi_impostor < lo_genuine:                      # clean gap -> take its middle
        t = 0.5 * (hi_impostor + lo_genuine)
        t = max(t, min(MIN_THRESHOLD, lo_genuine))
    else:                                             # overlap -> best balanced accuracy
        grid = np.arange(0.0, 1.0001, 0.01)
        bal = [0.5 * ((genuine >= g).mean() + (impostor < g).mean()) for g in grid]
        t = float(grid[int(np.argmax(bal))])
    t = round(t, 2)
    bal_acc = 0.5 * ((genuine >= t).mean() + (impostor < t).mean())
    return t, genuine, impostor, float(bal_acc)


def stranger_scores(engine, centres, folder):
    """Best cosine to any member centre for each stranger photo in folder."""
    scores = []
    for p in list_images(folder):
        img = cv2.imread(p)
        e = engine.largest_face_embedding(img) if img is not None else None
        if e is not None:
            scores.append(float((centres @ e).max()))
    return np.array(scores, dtype=np.float32)


def predict(centres, threshold, emb):
    cos = centres @ emb
    j = int(np.argmax(cos))
    return (j if cos[j] >= threshold else -1), float(cos[j])


def save_model(path, members, centres, threshold, meta):
    import torch
    torch.save({"members": members, "centres": torch.from_numpy(centres),
                "threshold": threshold, "s": ARC_S, "m": ARC_M, **meta}, path)


def load_model(path=MODEL_PATH):
    import torch
    if not os.path.exists(path):
        sys.exit(f"Model file not found: {path}\nRun:  python Group_19_1.py train")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["members"], ckpt["centres"].numpy(), float(ckpt["threshold"]), ckpt


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
GREEN, RED, WHITE, BLACK = (0, 200, 0), (0, 0, 230), (255, 255, 255), (0, 0, 0)


def draw_label(frame, box, text, colour):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = y1 - 8 if y1 - th - 12 > 0 else y2 + th + 8
    cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 8, y_text + 4), colour, -1)
    cv2.putText(frame, text, (x1 + 4, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)


def annotate(frame, engine, members, centres, threshold, show_score=True):
    """Detect, recognise and draw every face in the frame. Returns list of (name, score)."""
    results = []
    for box, kps, _ in engine.detect(frame):
        label, score = predict(centres, threshold, engine.embed(frame, kps))
        if label >= 0:
            name, colour = display_name(members[label]), GREEN
        else:
            name, colour = "Unidentified", RED
        text = f"{name} ({score:.2f})" if show_score else name
        draw_label(frame, box, text, colour)
        results.append((name, score))
    return results


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened() and os.name == "nt":            # Windows fallback
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"Could not open webcam {index}. Try --camera 1")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_capture(args):
    """Save webcam photos of one member into captures/<name>/sample_N.jpg."""
    out_dir = os.path.join(CAPTURE_DIR, args.name)
    os.makedirs(out_dir, exist_ok=True)
    existing = list_images(out_dir)
    n = max([sample_number(os.path.basename(p)) for p in existing], default=0)
    target = n + args.count

    engine = FaceEngine(det_size=320)
    cap = open_camera(args.camera)
    capturing, last_save = False, 0.0
    print(f"\nCapturing for '{args.name}' ({len(existing)} photos already saved).")
    print("SPACE = start/pause auto-capture   Q = quit")
    print("Tip: move your head, change distance, expression, glasses; "
          "repeat this command in other places / lighting.\n")

    while n < target:
        ok, frame = cap.read()
        if not ok:
            break
        faces = engine.detect(frame)
        view = frame.copy()
        for box, _, _ in faces:
            colour = GREEN if len(faces) == 1 else RED
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(view, (x1, y1), (x2, y2), colour, 2)

        now = time.time()
        if capturing and len(faces) == 1 and now - last_save >= args.interval:
            n += 1
            cv2.imwrite(os.path.join(out_dir, f"sample_{n}.jpg"), frame)   # clean frame
            last_save = now

        status = "CAPTURING" if capturing else "PAUSED (press SPACE)"
        if len(faces) != 1:
            status += "  - need exactly ONE face in view"
        cv2.putText(view, f"{args.name}: {n}/{target}  {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 4)
        cv2.putText(view, f"{args.name}: {n}/{target}  {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.imshow("Capture - Group 19", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            capturing = not capturing
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved photos up to sample_{n}.jpg in {out_dir}")


def cmd_split(args):
    """
    Copy captures/<member>/ into data/train|validate|test/<member>/.
    Consecutive webcam frames are nearly identical, so we split in BLOCKS of
    consecutive photos (not single photos). Otherwise near-duplicates would
    sit in both train and test and inflate the test accuracy.
    """
    members = list_members(CAPTURE_DIR)
    if not members:
        sys.exit(f"No member folders in {CAPTURE_DIR}. Run the capture command first.")
    ratios = np.array(args.ratios, dtype=float)
    ratios = ratios / ratios.sum()

    if os.path.isdir(DATA_DIR):
        if not args.force:
            sys.exit("data/ already exists. Use --force to rebuild it (it will be overwritten).")
        shutil.rmtree(DATA_DIR)

    rng = random.Random(SEED)
    print(f"Splitting with ratios train/validate/test = {np.round(ratios, 2).tolist()}, "
          f"block size {args.block}")
    for member in members:
        paths = list_images(os.path.join(CAPTURE_DIR, member))
        blocks = [paths[i:i + args.block] for i in range(0, len(paths), args.block)]
        rng.shuffle(blocks)
        n_blocks = len(blocks)
        n_val = max(1, int(round(ratios[1] * n_blocks)))
        n_test = max(1, int(round(ratios[2] * n_blocks)))
        n_train = n_blocks - n_val - n_test
        if n_train < 1:
            sys.exit(f"{member}: too few photos ({len(paths)}) to split. Capture more.")
        assignment = (["train"] * n_train + ["validate"] * n_val + ["test"] * n_test)
        counts = {s: 0 for s in SPLITS}
        for split, block in zip(assignment, blocks):
            dst = os.path.join(DATA_DIR, split, member)
            os.makedirs(dst, exist_ok=True)
            for p in block:
                shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
                counts[split] += 1
        print(f"   {member:<22} train {counts['train']:4d}  validate {counts['validate']:4d}  "
              f"test {counts['test']:4d}")
    print(f"Done -> {DATA_DIR}")


def cmd_train(args):
    set_seed()
    members = list_members(os.path.join(DATA_DIR, "train"))
    if len(members) < 2:
        sys.exit("Need at least 2 member folders in data/train/.")
    if list_members(os.path.join(DATA_DIR, "validate")) != members:
        sys.exit("data/validate/ must contain the same member folders as data/train/.")
    print(f"Members ({len(members)}): {', '.join(members)}")

    engine = FaceEngine(det_size=640)
    print("\n[1/4] Embedding training photos (with augmentation)...")
    Xtr, ytr = embed_folder_set(engine, os.path.join(DATA_DIR, "train"), members, n_aug=args.aug)
    print("\n[2/4] Embedding validation photos...")
    Xva, yva = embed_folder_set(engine, os.path.join(DATA_DIR, "validate"), members)
    if len(Xtr) == 0 or len(Xva) == 0:
        sys.exit("No faces found in train or validate photos.")

    print(f"\n[3/4] Training ArcFace head (s={ARC_S}, m={ARC_M}) "
          f"on {len(Xtr)} train / {len(Xva)} validation embeddings...")
    head, history = train_head(Xtr, ytr, Xva, yva, len(members), epochs=args.epochs, lr=args.lr)
    centres = class_centres(head)

    print("\n[4/4] Tuning the 'Unidentified' threshold on the validation set...")
    cos_va = Xva @ centres.T
    strangers = stranger_scores(engine, centres, os.path.join(UNKNOWN_DIR, "validate"))
    if len(strangers):
        print(f"   using {len(strangers)} stranger photos from unknown/validate/")
    else:
        print("   no unknown/validate/ photos -> using leave-one-out impostors only "
              "(adding stranger photos is recommended)")
    threshold, genuine, impostor, bal_acc = choose_threshold(cos_va, yva, strangers)
    if args.threshold is not None:
        print(f"   tuned value {threshold:.2f} overridden by --threshold {args.threshold:.2f}")
        threshold = args.threshold
    print(f"   genuine  cosine: mean {genuine.mean():.3f}  min {genuine.min():.3f}")
    print(f"   impostor cosine: mean {impostor.mean():.3f}  max {impostor.max():.3f}")
    print(f"   threshold = {threshold:.2f}   (balanced accuracy on validation {bal_acc * 100:.2f}%)")

    val_closed = (cos_va.argmax(1) == yva).mean()
    print(f"\nValidation accuracy (closed set, members only): {val_closed * 100:.2f}%")

    save_model(MODEL_PATH, members, centres, threshold,
               {"history": history, "train_size": int(len(Xtr)), "val_size": int(len(Xva))})
    print(f"Model saved -> {MODEL_PATH}")
    save_training_plot(history, genuine, impostor, threshold)


def save_training_plot(history, genuine, impostor, threshold):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    h = np.array(history)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h[:, 0], label="train loss")
    ax[0].plot(h[:, 1], label="validation loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("ArcFace loss"); ax[0].legend()
    ax[0].set_title("ArcFace head training")
    bins = np.linspace(-0.2, 1.0, 49)
    ax[1].hist(genuine, bins=bins, alpha=0.7, label="genuine (own class)")
    ax[1].hist(impostor, bins=bins, alpha=0.7, label="impostor (other members + strangers)")
    ax[1].axvline(threshold, color="k", linestyle="--", label=f"threshold {threshold:.2f}")
    ax[1].set_xlabel("cosine similarity to class centre"); ax[1].legend()
    ax[1].set_title("Validation scores")
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "Group_19_training.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Training plot saved -> {out}")


def cmd_test(args):
    members, centres, threshold, _ = load_model()
    if args.threshold is not None:
        threshold = args.threshold
    engine = FaceEngine(det_size=640)
    print(f"Threshold = {threshold:.2f}\n\nEmbedding test photos...")
    Xte, yte = embed_folder_set(engine, os.path.join(DATA_DIR, "test"), members)
    if len(Xte) == 0:
        sys.exit("No test faces found in data/test/.")

    cos = Xte @ centres.T
    closed_pred = cos.argmax(1)
    best = cos.max(1)
    open_pred = np.where(best >= threshold, closed_pred, -1)

    n = len(members)
    print(f"\nTest samples: {len(yte)}")
    print(f"Closed-set accuracy (argmax only)         : {(closed_pred == yte).mean() * 100:6.2f}%")
    print(f"Open-set accuracy (with 'Unidentified')   : {(open_pred == yte).mean() * 100:6.2f}%")
    print(f"Members wrongly rejected as Unidentified  : {(open_pred == -1).mean() * 100:6.2f}%")

    masked = cos.copy()
    masked[np.arange(len(yte)), yte] = -np.inf
    print(f"Leave-one-out unknowns correctly rejected : "
          f"{(masked.max(1) < threshold).mean() * 100:6.2f}%  "
          "(each member treated as if not enrolled)")

    # confusion matrix (rows = true, cols = predicted, last col = Unidentified)
    cm = np.zeros((n, n + 1), dtype=int)
    for t, p in zip(yte, open_pred):
        cm[t, p if p >= 0 else n] += 1
    w = max(12, max(len(m) for m in members) + 1)
    print("\nConfusion matrix (rows = true, columns = predicted)")
    header = " " * w + "".join(f"{m[:10]:>12}" for m in members) + f"{'Unidentif.':>12}"
    print(header)
    for i, m in enumerate(members):
        print(f"{m:<{w}}" + "".join(f"{v:>12d}" for v in cm[i]))

    s = stranger_scores(engine, centres, os.path.join(UNKNOWN_DIR, "test"))
    if len(s):
        rejected = int((s < threshold).sum())
        print(f"\nReal strangers in unknown/test/: {rejected}/{len(s)} correctly shown as "
              f"Unidentified ({rejected / len(s) * 100:.2f}%), highest score {s.max():.2f}")
    else:
        print("\n(Optional) put photos of people NOT in the group into unknown/test/ "
              "to measure 'Unidentified' accuracy on real strangers.")


def cmd_run(args):
    members, centres, threshold, _ = load_model()
    if args.threshold is not None:
        threshold = args.threshold
    engine = FaceEngine(det_size=args.det_size)
    cap = open_camera(args.camera)
    print(f"Members: {', '.join(display_name(m) for m in members)}   threshold {threshold:.2f}")
    print("Q = quit   S = toggle scores   + / - = raise / lower threshold")

    show_score, fps, t_prev = True, 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame not received.")
            break
        annotate(frame, engine, members, centres, threshold, show_score)

        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        info = f"ArcFace | FPS {fps:4.1f} | threshold {threshold:.2f}"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 4)
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        cv2.imshow("Group 19 - Face Recognition (ArcFace)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            show_score = not show_score
        elif key in (ord("+"), ord("=")):
            threshold = min(1.0, threshold + 0.02)
        elif key in (ord("-"), ord("_")):
            threshold = max(0.0, threshold - 0.02)

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Group 19 - ArcFace face recognition")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("capture", help="capture webcam photos of one member")
    c.add_argument("--name", required=True, help="member folder name, e.g. Aditya_Gajjar")
    c.add_argument("--count", type=int, default=60, help="photos to add in this session")
    c.add_argument("--interval", type=float, default=0.3, help="seconds between saved photos")
    c.add_argument("--camera", type=int, default=0)

    s = sub.add_parser("split", help="split captures/ into data/train|validate|test")
    s.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    s.add_argument("--block", type=int, default=10, help="consecutive photos kept together")
    s.add_argument("--force", action="store_true", help="overwrite an existing data/ folder")

    t = sub.add_parser("train", help="train the ArcFace head and tune the threshold")
    t.add_argument("--aug", type=int, default=2, help="augmented copies per training photo")
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--lr", type=float, default=0.01)
    t.add_argument("--threshold", type=float, default=None, help="override tuned threshold")

    e = sub.add_parser("test", help="evaluate on data/test (and unknown/ if present)")
    e.add_argument("--threshold", type=float, default=None)

    r = sub.add_parser("run", help="live webcam recognition")
    r.add_argument("--camera", type=int, default=0)
    r.add_argument("--det-size", type=int, default=320, help="detector input size (320 = faster)")
    r.add_argument("--threshold", type=float, default=None)

    args = p.parse_args()
    {"capture": cmd_capture, "split": cmd_split, "train": cmd_train,
     "test": cmd_test, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
