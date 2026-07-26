# Sleep Doctor 🩺🌙

**AI4ALL Ignite portfolio project** — predicting sleep quality from lifestyle factors.

**🔴 Live demo: [sleep-doctor-ai4all.streamlit.app](https://sleep-doctor-ai4all.streamlit.app)**

> **Research question:** Based on age, stress level, and activity level, can we predict sleep quality?

**Team:** Quang Doan · Shaili Halani · Nathanael Owusu · Prevailer Nchekwube · Sanskriti Poudel · Alex Saidov

## The project

We train two models on the [Sleep Health dataset](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset) (100,000 records, synthetic):

- **Path A — Linear Regression:** predicts the sleep quality score (1–10), then buckets it into Low / Medium / High
- **Path B — Random Forest:** classifies Low / Medium / High directly

We compare both paths, validate with k-fold cross-validation, interpret with permutation importance, and audit fairness across occupations and gender. **Gradient Boosting** joins later as a third challenger — not to crown a winner, but to test whether *any* algorithm can beat the two we proposed. A follow-up **extension analysis** then adds the rest of a person's lifestyle/context (work hours, shift work, caffeine, chronotype, mental health…) to see how far the ceiling moves.

Finally, an interactive **Streamlit app** lets anyone try the models live.

## What we found

- **Of the three research-question factors, only stress matters.** Age and activity add essentially nothing — stress alone predicts sleep-quality class about as well as all three combined.
- **Both of our models clear the 44.5% majority-class baseline, and then tie with each other (~61%).** Properly tuned, Linear Regression and Random Forest finish a tenth of a point apart — evidence the signal is one clean linear stress effect, with no hidden non-linear structure for a tree ensemble to find.
- **The extension lifts accuracy to ~68%** (R² 0.41 → 0.60) by adding lifestyle/context factors — shift work (−1.0 pt), mental health (~0.9 pt), and weekends (~0.7 pt) are the largest new effects.
- **~40% of sleep quality stays unexplained by lifestyle entirely** — an honest limit, and the data is synthetic, so these are patterns in the generator, not verified facts about human sleep.

### Only stress carries signal

Sleep quality falls steadily as stress rises, while age and step count produce flat clouds — the same three-panel comparison we put in front of the team before choosing features.

![Sleep quality against stress, age, and daily steps](figures/03_signal_vs_no_signal.png)

![Average sleep quality by stress level](figures/05_avg_quality_by_stress.png)

### What we predict

Rounding each score to the nearest integer and cutting at ≤4 / 5–6 / ≥7 gives the three classes. Medium is the largest at 44.5%, which is the number every model has to beat.

![Distribution of sleep quality scores by class](figures/01_sleep_quality_histogram.png)

### The result: our two models, on the question we asked

This is the project as proposed — Path A against Path B, on age, stress, and activity, judged on the held-out test set.

| Model | Held-out accuracy | R² |
|---|---|---|
| Baseline (always guess Medium) | 44.5% | — |
| **Path A — Linear Regression** | **61.2%** | 0.41 |
| **Path B — Random Forest** (`max_depth=8`) | **61.3%** | — |

Both beat the baseline by about 17 points, and they finish within a tenth of a point of each other. The tie is the interesting part: a tree ensemble given free rein to find curves and interactions can't do better than a straight line, which says the relationship really is one clean linear stress effect.

### Two follow-up questions we then asked

Neither of these replaces the comparison above — they're checks we ran afterwards to find out whether we'd left anything on the table.

**Could a more powerful algorithm beat them?** No. We added Gradient Boosting purely as a challenger and it reached 61.8%\* — inside a point of both proposed models. When three different algorithm families land in the same place, the ceiling belongs to the features, not the model.

**Could more of a person's day explain more?** Yes, by about 7 points. Adding lifestyle and context features (work hours, shift work, caffeine, screen time, chronotype, mental health, occupation) lifts Linear Regression to 68.3% and Gradient Boosting to 68.8%, with R² rising from 0.41 to 0.60. Random Forest reached 67.9%\* on the same features. This widens the original research question, so we report it as an extension rather than folding it into the headline result.

\* Validation-set figure. Each *final* model visits the test set exactly once — we report validation scores for models we didn't deploy rather than spending the test set on them.

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
| `notebooks/05_model_refinements.ipynb` | 4.5 | 70/15/15 split, forest depth tuning, the Gradient Boosting challenger, and the lifestyle/context extension |

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

## Limitations

- **The data is synthetic.** Every relationship here is a pattern a generator was programmed to produce. Nothing in this repo is evidence about human sleep — it is evidence about how well our methods recover structure that we know is present.
- **Most of sleep quality is missing from these features.** At R² 0.60 even the extension model leaves ~40% of the variation unexplained, and the research-question model leaves ~59%.
- **The High class is rare and poorly served.** Good sleepers are only 14% of the data, and recall on that class is the weakest part of every model — the people our tool would most like to identify are the ones it identifies worst.
- **Accuracy is not evenly distributed.** Our fairness audit found the models perform measurably better for some occupations than others, so a per-person prediction deserves less confidence than the headline number implies.
- **This is a wellness demo, not a diagnostic.** Nothing here should inform a medical decision.

## Next steps

- Validate the findings on real, non-synthetic sleep data (CDC NHANES and BRFSS both carry sleep items) — the single most valuable thing this project could do next.
- Add the measured-sleep features (duration, latency, wake episodes) as a separate "estimate quality from wearable data" model, which our experiments put near R² 0.77.
- Try models built for ordered categories, since Low / Medium / High is an ordinal outcome that both of our paths currently treat as unordered.
- Close the fairness gap surfaced by the occupation audit before this could be defensible in any real deployment.

## How we work

- `main` is protected on GitHub — direct pushes are rejected, so every change arrives by pull request
- One branch per phase: `feat/phase1-data-setup`, `feat/phase2-statistics`, …
- Commit style: `feat:` / `fix:` / `docs:` / `chore:`
- Every change is proposed in a pull request and reviewed before merge

## References

**Data**

1. Thalla, M. *Sleep Health & Daily Performance Dataset.* Kaggle. https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset

**Sleep science** — background for the relationships we tested

2. Åkerstedt, T. (2006). Psychosocial stress and impaired sleep. *Scandinavian Journal of Work, Environment & Health*, 32(6), 493–501. https://doi.org/10.5271/sjweh.1054
3. Kim, E. J., & Dimsdale, J. E. (2007). The effect of psychosocial stress on sleep: A review of polysomnographic evidence. *Behavioral Sleep Medicine*, 5(4), 256–278. https://doi.org/10.1080/15402000701557383
4. Boivin, D. B., & Boudreau, P. (2014). Impacts of shift work on sleep and circadian rhythms. *Pathologie Biologie*, 62(5), 292–301. https://doi.org/10.1016/j.patbio.2014.08.001

**Tools**

5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
6. McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. https://doi.org/10.25080/Majora-92bf1922-00a
7. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55
8. Streamlit documentation. https://docs.streamlit.io

**Ethics and bias**

9. Dastin, J. (2018, October 10). Amazon scraps secret AI recruiting tool that showed bias against women. *Reuters*. https://www.reuters.com/article/us-amazon-com-jobs-automation-insight-idUSKCN1MK08G
