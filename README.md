# Sleep Doctor 🩺🌙

**AI4ALL Ignite portfolio project** — predicting sleep quality from lifestyle factors.

**🔴 Live demo: [sleep-doctor-ai4all.streamlit.app](https://sleep-doctor-ai4all.streamlit.app)**

> **Research question:** Based on age, stress level, and activity level, can we predict sleep quality?

**Team:** Quang Doan · Shaili Halani · Nathanael Owusu · Prevailer Nchekwube · Sanskriti Poudel · Alex Saidov

## The project

We train two models on the [Sleep Health dataset](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset) (100,000 records, synthetic):

- **Path A — Linear Regression:** predicts the sleep quality score (1–10), then buckets it into Low / Medium / High
- **Path B — Random Forest:** classifies Low / Medium / High directly

We compare both paths, validate with k-fold cross-validation, interpret with permutation importance, and audit fairness across occupations and gender. A follow-up **extension analysis** then adds the rest of a person's lifestyle/context (work hours, shift work, caffeine, chronotype, mental health…) to see how far the ceiling moves.

Finally, an interactive **Streamlit app** lets anyone try the models live.

## What we found

- **Of the three research-question factors, only stress matters.** Age and activity add essentially nothing — stress alone predicts sleep-quality class about as well as all three combined.
- **Both models clear the 44.5% majority-class baseline;** properly tuned, Linear Regression and Random Forest tie (~61%), evidence the signal is one clean linear stress effect with no hidden non-linear structure.
- **The extension lifts accuracy to ~68%** (R² 0.41 → 0.60) by adding lifestyle/context factors — shift work (−1.0 pt), mental health (~0.9 pt), and weekends (~0.7 pt) are the largest new effects.
- **~40% of sleep quality stays unexplained by lifestyle entirely** — an honest limit, and the data is synthetic, so these are patterns in the generator, not verified facts about human sleep.

| Model | Features | Held-out accuracy | R² |
|---|---|---|---|
| Baseline (always guess Medium) | — | 44.5% | — |
| Research question | age, stress, activity | 61% | 0.41 |
| Extension | + full lifestyle/context | 68% | 0.60 |

## Getting started

```bash
git clone https://github.com/Poudel-Sanskriti/Sleep_Doctor.git
cd Sleep_Doctor
pip install -r requirements.txt
```

### Run the interactive app

```bash
streamlit run app.py
```

Pick a mode (research question vs. full lifestyle), move the sliders, and see the predicted sleep-quality score, its Low/Medium/High class, and a per-input breakdown of what drove the prediction.

### Explore the analysis notebooks

```bash
jupyter notebook
```

Open the notebooks in order — each one is a phase of the project:

| Notebook | Phase | What it does |
|---|---|---|
| `notebooks/01_setup_and_cleaning.ipynb` | 1 | Load the dataset, verify integrity, sanity-check values |
| `notebooks/02_statistics.ipynb` | 2 | Descriptive stats, correlations, t-test |
| `notebooks/03_visualization.ipynb` | 3 | The five key charts (saved to `figures/`) |
| `notebooks/04_models.ipynb` | 4 | Train, compare, and evaluate both models |
| `notebooks/05_model_refinements.ipynb` | 4.5 | 70/15/15 split, depth tuning, and the lifestyle/context extension |

## How the app works

The app follows a **train/serve split**, the way real ML systems separate fitting a model from using it:

| File | Role |
|---|---|
| `train.py` | Offline training. Fits both Linear Regression models, evaluates them, and exports coefficients + metrics to `models.json`. Run it whenever the data or features change. |
| `models.json` | The trained artifact the app ships. Small, human-readable — no pickle/version-lock headaches. |
| `model_config.py` | Feature schema shared by `train.py` and `app.py`, so serving can never drift from what was trained. |
| `app.py` | Loads `models.json` at startup and computes predictions — it never retrains. |

`train.py` uses the same 70/15/15 stratified split (seed 117) as `05_model_refinements.ipynb`, so the app's reported numbers match the notebook exactly.

To regenerate the artifact after changing the data or features:

```bash
python train.py
```

## How we work

- `main` is protected — all changes arrive by pull request
- One branch per phase: `feat/phase1-data-setup`, `feat/phase2-statistics`, …
- Commit style: `feat:` / `fix:` / `docs:` / `chore:`
- Every PR gets reviewed by a teammate before merge
