import numpy as np
import librosa
import csv
from pathlib import Path
from tqdm import tqdm

# =========================
# PARAMETERS
# =========================

SAMPLING_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
F_MIN = 0
F_MAX = 8000

# =========================
# PATHS
# =========================

DATASET_ROOT = Path(__file__).resolve().parent / "dataset" / "voxceleb-20261023T105654Z-1-001"

WAV_ROOT = DATASET_ROOT / "voxceleb" / "wavs"
CSV_PATH = DATASET_ROOT / "voxceleb_vectors33.csv"

# =========================
# AUDIO → VECTOR
# =========================

def wav_to_feature_vector(wav_path: Path):
    try:
        y, sr = librosa.load(str(wav_path), sr=SAMPLING_RATE, mono=True, duration=3)
    except Exception:
        y = np.zeros(SAMPLING_RATE, dtype=np.float32)

    if y is None or len(y) == 0:
        y = np.zeros(SAMPLING_RATE, dtype=np.float32)

    if len(y) < SAMPLING_RATE:
        y = np.pad(y, (0, SAMPLING_RATE - len(y)))

    # Normalize
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    # Trim silence
    try:
        y, _ = librosa.effects.trim(y, top_db=30)
    except Exception:
        pass

    if y is None or len(y) == 0:
        y = np.random.randn(SAMPLING_RATE) * 1e-6

    if len(y) < SAMPLING_RATE // 2:
        y = np.pad(y, (0, (SAMPLING_RATE // 2) - len(y)))

    # Mel spectrogram
    try:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SAMPLING_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            fmin=F_MIN,
            fmax=F_MAX,
            power=2.0,
        )
    except Exception:
        mel = np.zeros((N_MELS, 10), dtype=np.float32)

    if mel is None or mel.size == 0:
        mel = np.zeros((N_MELS, 10), dtype=np.float32)

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # =========================
    # VECTOR (Mean + Std)
    # =========================
    mean = np.mean(mel_db, axis=1)
    std = np.std(mel_db, axis=1)

    feature_vector = np.concatenate([mean, std])  # (160,)

    return feature_vector.astype(np.float32)

# =========================
# PROCESS → CSV
# =========================

def process_voxceleb_to_csv():

    wav_files = sorted(WAV_ROOT.rglob("*.wav"))

    if not wav_files:
        print(f"[WARN] No .wav files found: {WAV_ROOT}")
        return

    print(f"\nSource : {WAV_ROOT}")
    print(f"CSV    : {CSV_PATH}")
    print(f"Files  : {len(wav_files)}")

    errors = 0

    # Create CSV
    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = [f"f{i}" for i in range(160)] + ["label"]
        writer.writerow(header)

        for wav_path in tqdm(wav_files, desc="Processing", unit="file"):

            try:
                rel_parts = wav_path.relative_to(WAV_ROOT).parts

                # Speaker label
                speaker_id = rel_parts[0]

                # Feature vector
                vec = wav_to_feature_vector(wav_path)

                # Write row
                row = vec.tolist() + [speaker_id]
                writer.writerow(row)

            except Exception as e:
                print(f"\n[ERROR] {wav_path} -> {e}")
                errors += 1

    print(f"\nDone ✅ | Errors: {errors}")

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    process_voxceleb_to_csv()