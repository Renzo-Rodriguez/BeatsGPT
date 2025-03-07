import os
import librosa
import numpy as np
import multiprocessing as mp
from glob import glob
from tqdm import tqdm
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_path):
    """Converts MP3 to WAV for librosa processing."""
    audio = AudioSegment.from_mp3(mp3_path)
    wav_path = mp3_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

def process_segment(segment, sr):
    """Extracts the top 3 harmonic frequencies for a segment."""
    if len(segment) == 0:
        return None

    fft_result = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), d=1/sr)
    top_freqs = freqs[np.argsort(fft_result)[-3:]]  # Top 3 frequencies
    return tuple(top_freqs)

def process_audio_file(file_path, segment_length=1/8, sample_rate=44100):
    """Processes an MP3 file, converting it to WAV and extracting frequency tokens."""
    try:
        # Convert MP3 to WAV if needed
        if file_path.endswith(".mp3"):
            file_path = convert_mp3_to_wav(file_path)

        y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        samples_per_segment = int(segment_length * sr)
        segments = [y[i:i+samples_per_segment] for i in range(0, len(y), samples_per_segment)]
        
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(process_segment, [(seg, sr) for seg in segments])

        return np.array([res for res in results if res is not None])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_folder(folder_path, output_folder, folder_index, total_folders):
    """Processes all MP3 files in a folder and saves tokenized data with a filename dictionary."""
    folder_name = os.path.basename(folder_path)
    audio_files = sorted(glob(os.path.join(folder_path, "*.mp3")))  # Get all MP3 files

    print(f"Processing folder {folder_index + 1}/{total_folders}: {folder_name} ({len(audio_files)} files)")

    token_dict = {}  # Dictionary to store filename → tokens

    for i, audio_file in enumerate(tqdm(audio_files, desc=f"Folder {folder_index + 1}/{total_folders}", unit="file")):
        tokens = process_audio_file(audio_file)
        if tokens is not None:
            token_dict[os.path.basename(audio_file)] = tokens  # Store with filename

    # Save processed tokens in structured format
    if token_dict:
        save_path = os.path.join(output_folder, f"{folder_name}_tokens.npz")
        np.savez_compressed(save_path, **token_dict)
        print(f"✔ Saved {folder_name} tokens to {save_path}")

def process_all_folders(base_folder, output_folder, num_workers=4):
    """Processes multiple folders in parallel, storing results in a dictionary format."""
    folders = sorted([f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))])
    total_folders = len(folders)

    os.makedirs(output_folder, exist_ok=True)

    with mp.Pool(num_workers) as pool:
        pool.starmap(process_folder, [(os.path.join(base_folder, folder), output_folder, idx, total_folders) for idx, folder in enumerate(folders)])

# Example usage
base_folder = "/home/renzo/projects/stempalooza/audio"
output_folder = "/home/renzo/projects/stempalooza/processed"
print(process_audio_file("/home/renzo/projects/stempalooza/audio/000/000002"))
#process_all_folders(base_folder, output_folder, num_workers=4)
