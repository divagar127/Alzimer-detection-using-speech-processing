"""
Main Pipeline Runner for Alzheimer's Detection using Speech Processing
Orchestrates:
Stage 1: Preprocessing (raw audio -> 16kHz, noise reduced, VAD segmented -> processed_audio/)
Stage 2: Feature Extraction (123 Acoustic Features + Deep Embeddings)
Stage 3: Feature Fusion (Concatenate + Dimensionality reduction to ~318 dimensions)
Stage 4: Classification (SVM, Random Forest, Logistic Regression, PyTorch DNN, PyTorch 1DCNN)
Stage 5: Evaluation (5-fold Stratified CV, ROC curves, Confusion Matrices, Metric Tables)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing import AudioPreprocessor
from src.feature_extraction import FeatureExtractionPipeline
from src.feature_fusion import FeatureFusion
from src.evaluation import Evaluator


def load_dataset_metadata(base_dir):
    """
    Find audio files and extract class labels without modifying any dataset files.
    :param base_dir: Root project directory
    :return: (file_paths, labels)
    """
    file_paths = []
    labels = []

    # 1. Diagnosis task: AD (1) vs CN (0)
    ad_dir = os.path.join(base_dir, "diagnosis_train", "train", "audio", "ad")
    cn_dir = os.path.join(base_dir, "diagnosis_train", "train", "audio", "cn")

    if os.path.exists(ad_dir):
        ad_files = glob.glob(os.path.join(ad_dir, "*.wav"))
        file_paths.extend(ad_files)
        labels.extend([1] * len(ad_files))

    if os.path.exists(cn_dir):
        cn_files = glob.glob(os.path.join(cn_dir, "*.wav"))
        file_paths.extend(cn_files)
        labels.extend([0] * len(cn_files))

    # 2. Progression task: decline (1) vs no_decline (0) if diagnosis not present or combined
    if len(file_paths) == 0:
        prog_decline = os.path.join(base_dir, "progression_train", "ADReSSo21", "progression", "train", "audio", "decline")
        prog_no_decline = os.path.join(base_dir, "progression_train", "ADReSSo21", "progression", "train", "audio", "no_decline")

        if os.path.exists(prog_decline):
            dec_files = glob.glob(os.path.join(prog_decline, "*.wav"))
            file_paths.extend(dec_files)
            labels.extend([1] * len(dec_files))

        if os.path.exists(prog_no_decline):
            nodec_files = glob.glob(os.path.join(prog_no_decline, "*.wav"))
            file_paths.extend(nodec_files)
            labels.extend([0] * len(nodec_files))

    return np.array(file_paths), np.array(labels)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "processed_audio")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print("==========================================================================", flush=True)
    print("  ALZHEIMER'S DETECTION USING SPEECH PROCESSING - PIPELINE INITIALIZATION", flush=True)
    print("==========================================================================", flush=True)

    # 0. Load Dataset Metadata
    raw_files, labels = load_dataset_metadata(base_dir)
    print(f"\n[Dataset Metadata] Found {len(raw_files)} audio samples (AD/Decline: {np.sum(labels == 1)}, CN/No-Decline: {np.sum(labels == 0)})", flush=True)

    if len(raw_files) == 0:
        print("Error: No .wav audio files found in expected dataset directories.", flush=True)
        return

    # STAGE 1: DATA PREPROCESSING
    print("\n>>> STAGE 1: DATA PREPROCESSING (Resampling to 16kHz, Noise Reduction, VAD Silence Removal)", flush=True)
    preprocessor = AudioPreprocessor(target_sr=16000, top_db=25, reduce_noise=True)
    processed_files = []

    for raw_path in tqdm(raw_files, desc="Preprocessing Audio Files"):
        rel_path = os.path.relpath(raw_path, base_dir)
        out_path = os.path.join(processed_dir, rel_path)
        # Check if already preprocessed to save time
        if not os.path.exists(out_path):
            preprocessor.process_file(raw_path, out_path)
        processed_files.append(out_path)

    print(f"Preprocessed audio files saved to: {processed_dir}", flush=True)

    # STAGE 2: FEATURE EXTRACTION
    print("\n>>> STAGE 2: FEATURE EXTRACTION (123 Acoustic Features + Deep Embeddings)", flush=True)
    cache_file = os.path.join(results_dir, "features_cache.npz")
    
    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...", flush=True)
        data = np.load(cache_file)
        acoustic_feats = data["acoustic"]
        deep_feats = data["deep"]
        labels = data["labels"]
    else:
        fe_pipeline = FeatureExtractionPipeline(sr=16000)
        acoustic_list, deep_list = [], []

        for p_file in tqdm(processed_files, desc="Extracting Features"):
            y, sr = preprocessor.process_file(p_file)
            feats = fe_pipeline.extract_all(y)
            acoustic_list.append(feats["acoustic"]) # 123-dim
            deep_list.append(feats["whisper"])      # 1024-dim

        acoustic_feats = np.array(acoustic_list)
        deep_feats = np.array(deep_list)

        np.savez(cache_file, acoustic=acoustic_feats, deep=deep_feats, labels=labels)
        print(f"Features successfully cached to {cache_file}", flush=True)

    print(f"Acoustic features shape: {acoustic_feats.shape} (Expected: N x 123)", flush=True)
    print(f"Deep embeddings shape:   {deep_feats.shape} (Expected: N x 1024)", flush=True)

    # STAGE 3: FEATURE FUSION
    print("\n>>> STAGE 3: FEATURE FUSION & DIMENSIONALITY REDUCTION (~318 dimensions)", flush=True)
    fusion = FeatureFusion(target_dim=318, method="pca")
    concatenated_feats = fusion.concatenate_features(acoustic_feats, deep_feats)
    print(f"Concatenated features shape: {concatenated_feats.shape} (123 + 1024 = 1147)", flush=True)

    fused_features = fusion.fit_transform(concatenated_feats)
    print(f"Final Fused Features shape: {fused_features.shape} (Target ~318 dimensions)", flush=True)

    # STAGE 4 & 5: CLASSIFICATION & EVALUATION
    print("\n>>> STAGE 4 & 5: CLASSIFICATION & 5-FOLD CROSS-VALIDATION EVALUATION", flush=True)
    evaluator = Evaluator(output_dir=results_dir)
    df_summary, results = evaluator.evaluate_pipeline(fused_features, labels, n_splits=5)

    print("\n==========================================================================", flush=True)
    print("                     EVALUATION RESULTS COMPARISON                        ", flush=True)
    print("==========================================================================", flush=True)
    print(df_summary.to_string(index=False), flush=True)

    # Print Comparison with Baseline Papers Table
    print("\n==========================================================================", flush=True)
    print("           COMPARISON WITH BASELINE PAPERS (ADReSSo Challenge)            ", flush=True)
    print("==========================================================================", flush=True)
    baseline_table = pd.DataFrame([
        {"Method / Study": "Linguistic Baseline (BERT)", "Acoustic / Deep": "Text / Transcripts", "Accuracy (%)": "76.7%", "F1-Score": "0.765"},
        {"Method / Study": "Acoustic Baseline (eGeMAPS + SVM)", "Acoustic / Deep": "eGeMAPS (Acoustic)", "Accuracy (%)": "65.1%", "F1-Score": "0.640"},
        {"Method / Study": "Wav2Vec2 Fine-tuned (Papasavvas et al.)", "Acoustic / Deep": "Wav2Vec2", "Accuracy (%)": "78.2%", "F1-Score": "0.779"},
        {"Method / Study": "Our Proposed Pipeline (Fusion + SVM)", "Acoustic / Deep": "Acoustic (123) + Deep (1024)", "Accuracy (%)": df_summary[df_summary['Model']=='SVM (RBF)']['Accuracy (%)'].values[0], "F1-Score": df_summary[df_summary['Model']=='SVM (RBF)']['F1-Score'].values[0]},
        {"Method / Study": "Our Proposed Pipeline (Fusion + DNN)", "Acoustic / Deep": "Acoustic (123) + Deep (1024)", "Accuracy (%)": df_summary[df_summary['Model']=='DNN']['Accuracy (%)'].values[0], "F1-Score": df_summary[df_summary['Model']=='DNN']['F1-Score'].values[0]},
        {"Method / Study": "Our Proposed Pipeline (Fusion + 1DCNN)", "Acoustic / Deep": "Acoustic (123) + Deep (1024)", "Accuracy (%)": df_summary[df_summary['Model']=='1DCNN']['Accuracy (%)'].values[0], "F1-Score": df_summary[df_summary['Model']=='1DCNN']['F1-Score'].values[0]},
    ])
    print(baseline_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
