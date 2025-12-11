# Demosaicing Pattern Inconsistency Detector

Course project implementing a demosaicing-based image forensics method
(Gallagher & Chen style) to:
- distinguish camera images from non-demosaiced images,
- localize possible forgeries using local demosaicing-pattern inconsistency.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
