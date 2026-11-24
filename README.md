# An Image-Based Deep Learning Framework for Leprosy Diagnosis and Severity Assessment

Code and supplementary data for a three-stage cascade for skin-image analysis:

1. **Stage 1 – detection:** leprosy vs. non-leprosy, soft-voting ensemble of EfficientNet-B3, ConvNeXt-Tiny and ResNet-50.
2. **Stage 2 – subtype:** Tuberculoid / Lepromatous / Borderline, EfficientNet-B3.
3. **Stage 3 – severity band:** a fixed rule mapping the subtype to a WHO / Ridley-Jopling band (paucibacillary, intermediate, multibacillary). It is a lookup, not a trained model and not an independent severity measurement.

Grad-CAM maps are provided for the Stage 1 and Stage 2 decisions.

> **Research code only.** This is not a diagnostic device and must not be used for clinical decisions. Performance drops markedly on images from a source different from the training data (see Main results).

## Main results (hold-out and external)

| Evaluation | Result |
|---|---|
| Stage 1, internal test set (737 images) | accuracy 97.0%, AUC 0.981 (ensemble) |
| Stage 1, external set (Roboflow AI-Leprosy) | AUC 0.766, recall 33.3% at threshold 0.5 (ensemble) |
| Stage 2, clean hold-out (101 leprosy images, never used for training or model selection) | accuracy 84.2% (85/101); Borderline recall 52.6% (10/19) |

Run-to-run variation (retraining with different seeds) is as large as the differences between the ablated configurations; see the paper and `supp/ablation/`. The external gap is not closed by any variant tested.

## Repository contents

| File | Purpose |
|---|---|
| `Leprosy_Classification_FINAL.ipynb` | Stage 1: data organisation, training and evaluation of the three detectors and the ensemble |
| `Stage2_Leprosy_Subtype_Classification.ipynb` | Stage 2: subtype classifier, 5-fold CV on the pool and evaluation on the clean hold-out (uses `supp/splits/stage2_split_manifest.csv`) |
| `Stage1_Baseline_Ablation.ipynb`, `Stage1_Ablation_Runs.ipynb` | Seeds and ablations of the Stage 1 EfficientNet-B3 recipe |
| `External_Validation_Roboflow.ipynb` | External evaluation of Stage 1 on Roboflow AI-Leprosy, with leakage, threshold and error analyses |
| `Stage2_GradCAM_v2.ipynb` | Stage 2 Grad-CAM maps on hold-out images |
| `Stage2_Masking_Test.ipynb` | Masking test: Grad-CAM region vs. random region occlusion |
| `Leprosy_Diagnostic_System.ipynb` | Demo pipeline that runs the three stages on one image |
| `supp/` | Supplementary data (see `supp/README.md`): split manifests, per-image probabilities, ablation runs, leakage checks, masking-test results |

## Data

Images are **not** redistributed. Obtain them from the original sources under their licences:

- Training / internal: *31 classes of skin disease* (Atlas Dermatologico + ISIC 2019) – https://www.kaggle.com/datasets/kelixo/31-classes-of-skin-disease (ISIC: https://www.isic-archive.com/, Atlas: http://www.atlasdermatologico.com.br/)
- External validation: AI-Leprosy v1, Roboflow Universe (COCO export) – https://universe.roboflow.com/search?q=leprosy

`supp/splits/` lists exactly which image falls in which split (paths relative to the dataset root). Stage 1 split: 70/15/15 stratified, seed 42 (leprosy 470/101/101, non-leprosy 2965/636/636). Stage 2: hold-out = the 101 Stage 1 test leprosy images, pool = the 571 Stage 1 train + validation leprosy images, so no image of the hold-out is used for training or model selection.

Known limitation: patient identifiers are not available, so patient-level independence between splits cannot be verified.

## Reproducing

The notebooks were run on Google Colab (Tesla T4) with PyTorch 2.11.0 and torchvision 0.26.0.

1. Download the datasets and place them in Google Drive; edit the path constants at the top of each notebook.
2. Run in this order: Stage 1 → Stage 2 → external validation → Grad-CAM / masking test.
3. Seeds: 42 for the split and main runs; 43 and 44 for the repeats. Only the split seed is fixed globally, so retraining reproduces the reported numbers approximately, not exactly. The per-image outputs used for the reported tables are in `supp/`.

```bash
pip install -r requirements.txt
```

## Citation

If you use this work, please cite the paper (reference to be added when published) and the repository.

## License

MIT (code). The datasets keep their own licences.
