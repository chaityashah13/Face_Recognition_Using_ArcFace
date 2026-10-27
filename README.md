# Face_Recognition_Using_ArcFace

Mini project for the Deep Learning course. We compare DeepFace, FaceNet, VGG Face and ArcFace
(see `Group_19_Report.pdf`) and implement ArcFace for real-time recognition of our four members.

**Members:** Aditya Gajjar (KU2407U392), Henish Prajapati (KU2407U404),
Krish Dhairav (KU2407U416), Shah Chaitya (KU2407U503)

## Pipeline
1. SCRFD face detection (box + 5 landmarks)
2. 2D alignment to 112×112
3. Pretrained ArcFace ResNet-50 (InsightFace `buffalo_l`), frozen, giving a 512-d embedding
4. Our own ArcFace head trained on our members' embeddings (s = 64, m = 0.5)
5. Cosine similarity with a threshold, otherwise "Unidentified"

## Setup (Windows, Python 3.10/3.11, no GPU needed)
    python -m venv venv
    venv\Scripts\activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install "numpy<2"
    pip install -r requirements.txt

The first run downloads the `buffalo_l` model (~280 MB) automatically.

## Usage
    python Group_19_1.py capture --name First_Last
    python Group_19_1.py split
    python Group_19_1.py train
    python Group_19_1.py test
    python Group_19_1.py run

Face photos are not included in this repository for privacy reasons.
