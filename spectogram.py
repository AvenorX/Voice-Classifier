import numpy as np
import librosa
import cv2
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



WAV_ROOT = "voxceleb" / "wavs"
MEL_ROOT = "voxceleb" / "mel"

# =========================
# MEL FUNCTION
# =========================

def wav_to_mel_image(wav_path: Path):
    try:
        y, sr = librosa.load(str(wav_path), sr=SAMPLING_RATE, mono=True,duration=1)
    except Exception:
        # 🔥 fallback: 1 sec silence
        y = np.zeros(SAMPLING_RATE, dtype=np.float32)

    # Ensure valid signal
    if y is None or len(y) == 0:
        y = np.zeros(SAMPLING_RATE, dtype=np.float32)

    # Pad if too short
    if len(y) < SAMPLING_RATE:
        y = np.pad(y, (0, SAMPLING_RATE - len(y)))

    # Normalize safely
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    # Trim silence (safe)
    try:
        y, _ = librosa.effects.trim(y, top_db=30)
    except Exception:
        pass

    # If empty after trim → replace with small noise
    if y is None or len(y) == 0:
        y = np.random.randn(SAMPLING_RATE) * 1e-6

    # Ensure minimum length
    if len(y) < SAMPLING_RATE // 2:
        y = np.pad(y, (0, (SAMPLING_RATE // 2) - len(y)))

    # Mel spectrogram (safe)
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
        # 🔥 fallback mel (all zeros)
        mel = np.zeros((N_MELS, 10), dtype=np.float32)

    if mel is None or mel.size == 0:
        mel = np.zeros((N_MELS, 10), dtype=np.float32)

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = np.clip(mel_db, -80, 0)
    mel_norm = (mel_db + 80) / 80

    mel_img = (mel_norm * 255)

    if mel_img is None or mel_img.size == 0:
        mel_img = np.zeros((N_MELS, 10), dtype=np.uint8)

    mel_img = mel_img.astype(np.uint8)
    mel_img = np.flipud(mel_img)

    return mel_img

# =========================
# PROCESS VOXCELEB
# =========================

def process_voxceleb():

    wav_files = sorted(WAV_ROOT.rglob("*.wav"))

    if not wav_files:
        print(f"[WARN] No .wav files found: {WAV_ROOT}")
        return

    print(f"\nSource : {WAV_ROOT}")
    print(f"Output : {MEL_ROOT}")
    print(f"Files  : {len(wav_files)}")

    skipped = 0
    errors = 0

    for wav_path in tqdm(wav_files, desc="Processing", unit="file"):

        try:
            rel_parts = wav_path.relative_to(WAV_ROOT).parts

            # =========================
            # SPEAKER FOLDER
            # =========================
            speaker_id = rel_parts[0]

            save_dir = MEL_ROOT / speaker_id
            save_dir.mkdir(parents=True, exist_ok=True)

            # =========================
            # FILENAME (EXACT MATCH)
            # =========================
            filename = wav_path.stem + ".jpg"

            save_path = save_dir / filename

            # Skip if already exists
            if save_path.exists():
                skipped += 1
                continue

            # Generate mel
            mel_img = wav_to_mel_image(wav_path)

            # Save image
            success = cv2.imwrite(
                str(save_path),
                mel_img,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            if not success:
                print(f"\n[ERROR] Failed to save: {save_path}")
                errors += 1

        except Exception as e:
            print(f"\n[ERROR] {wav_path} -> {e}")
            errors += 1

    print(f"\nDone ✅ | Skipped: {skipped} | Errors: {errors}")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    process_voxceleb()