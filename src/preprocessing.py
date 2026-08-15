"""
Stage 1: Data Preprocessing Module
- Load .wav files (librosa)
- Noise reduction (noisereduce)
- Resampling to 16kHz
- Segmentation / Silence removal (VAD via librosa.effects.split)
- Save processed audio to output directory (leaves raw dataset untouched)
"""

import os
import glob
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from tqdm import tqdm


class AudioPreprocessor:
    def __init__(self, target_sr=16000, top_db=25, reduce_noise=True):
        """
        Initialize AudioPreprocessor.
        :param target_sr: Target sampling rate (default 16000 Hz)
        :param top_db: The threshold (in dB) below peak to consider as silence for VAD
        :param reduce_noise: Boolean flag to apply spectral gating noise reduction
        """
        self.target_sr = target_sr
        self.top_db = top_db
        self.reduce_noise = reduce_noise

    def process_file(self, input_path, output_path=None):
        """
        Process a single audio file: load, reduce noise, resample, remove silences, save.
        :param input_path: Path to raw input .wav file
        :param output_path: Optional path to save processed .wav file
        :return: (processed_audio_signal, sample_rate)
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file not found: {input_path}")

        # 1. Load raw audio file using librosa
        y, sr = librosa.load(input_path, sr=None)

        # Handle empty audio
        if len(y) == 0:
            y = np.zeros(self.target_sr, dtype=np.float32)
            sr = self.target_sr

        # 2. Resample to target sample rate (16kHz)
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        # 3. Noise reduction using noisereduce spectral gating
        if self.reduce_noise and len(y) > 0:
            try:
                y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.75)
            except Exception as e:
                # Fallback if noise reduction encounters edge cases
                pass

        # 4. Segmentation / VAD (Voice Activity Detection - remove silences)
        non_silent_intervals = librosa.effects.split(y, top_db=self.top_db)
        if len(non_silent_intervals) > 0:
            y_segmented = np.concatenate([y[start:end] for start, end in non_silent_intervals])
        else:
            y_segmented = y

        # Fallback if segmenting resulted in empty audio
        if len(y_segmented) == 0:
            y_segmented = y

        # 5. Normalize amplitude (-1.0 to 1.0)
        max_val = np.max(np.abs(y_segmented))
        if max_val > 0:
            y_segmented = y_segmented / max_val

        # 6. Save processed file if output_path is specified
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, y_segmented, self.target_sr)

        return y_segmented, self.target_sr

    def process_directory(self, input_dir, output_dir):
        """
        Process all .wav files in a directory hierarchy recursively and save to output_dir.
        :param input_dir: Directory containing raw audio files
        :param output_dir: Destination directory for processed audio files
        :return: List of tuples (processed_file_path, original_file_path)
        """
        wav_files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
        processed_files = []

        print(f"Found {len(wav_files)} .wav files in {input_dir}")
        for raw_path in tqdm(wav_files, desc="Preprocessing Audio Files"):
            rel_path = os.path.relpath(raw_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)
            self.process_file(raw_path, out_path)
            processed_files.append((out_path, raw_path))

        return processed_files


if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = glob.glob(os.path.join(base_dir, "**", "*.wav"), recursive=True)
    if test_file:
        preprocessor = AudioPreprocessor(target_sr=16000)
        out_test = os.path.join(base_dir, "processed_audio", "test_sample.wav")
        y_proc, sr = preprocessor.process_file(test_file[0], out_test)
        print(f"Successfully processed sample {test_file[0]} -> {out_test} (shape: {y_proc.shape}, sr: {sr})")
