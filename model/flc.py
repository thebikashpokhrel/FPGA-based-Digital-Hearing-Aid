import os
import soundfile as sf
import librosa
import numpy as np
from pydub import AudioSegment
import random

# Parameters
sample_rate = 16000  # 16kHz
duration = 2  # 2 seconds
input_voice_dir = './voice_samples'  # Update with your FLAC voice samples path
input_noise_dir = './noise_samples'  # Update with your WAV noise samples path
output_dir = './audio_dataset'
num_samples_per_class = 1000

# Create output directories
os.makedirs(f'{output_dir}/pure_voice', exist_ok=True)
os.makedirs(f'{output_dir}/line_noise', exist_ok=True)
os.makedirs(f'{output_dir}/noise_voice', exist_ok=True)
os.makedirs(f'{output_dir}/empty', exist_ok=True)

def convert_and_process_audio(input_path, output_path, is_flac=False):
    try:
        if is_flac:
            # Load FLAC with pydub
            audio = AudioSegment.from_file(input_path, format="flac")
            # Convert to mono, 16kHz
            audio = audio.set_channels(1).set_frame_rate(sample_rate)
            # Trim or pad to 2 seconds
            target_ms = duration * 1000
            if len(audio) > target_ms:
                audio = audio[:target_ms]
            else:
                audio = audio + AudioSegment.silent(duration=target_ms - len(audio))
            # Export as WAV
            audio.export(output_path, format="wav")
        else:
            # Load WAV with soundfile
            data, sr = sf.read(input_path)
            if sr != sample_rate:
                data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
            target_len = int(sample_rate * duration)
            if len(data) > target_len:
                data = data[:target_len]
            else:
                data = np.pad(data, (0, max(0, target_len - len(data))), mode='constant')
            # Normalize, handle zero or invalid cases
            max_abs = np.max(np.abs(data))
            if max_abs > 0 and not np.isnan(max_abs) and not np.isinf(max_abs):
                data = data / max_abs
            else:
                print(f"Warning: Invalid data in {input_path}. Replacing with low-amplitude noise.")
                data = np.random.normal(0, 0.01, len(data))  # Replace with noise
            sf.write(output_path, data, sample_rate, subtype='PCM_16')
    except Exception as e:
        print(f"Error processing {input_path}: {e}. Replacing with noise.")
        data = np.random.normal(0, 0.01, int(sample_rate * duration))
        sf.write(output_path, data, sample_rate, subtype='PCM_16')

def generate_empty():
    return np.random.normal(0, 0.01, int(sample_rate * duration))

def mix_audio(voice_path, noise_path):
    try:
        voice, sr = sf.read(voice_path)
        noise, _ = sf.read(noise_path)
        target_len = int(sample_rate * duration)
        voice = voice[:target_len]
        noise = noise[:target_len]
        mixed = 0.6 * voice + 0.4 * noise
        max_abs = np.max(np.abs(mixed))
        if max_abs > 0 and not np.isnan(max_abs) and not np.isinf(max_abs):
            mixed = mixed / max_abs
        else:
            print(f"Warning: Invalid mix for {voice_path} and {noise_path}. Using noise.")
            mixed = np.random.normal(0, 0.01, target_len)
        return mixed
    except Exception as e:
        print(f"Error mixing {voice_path} and {noise_path}: {e}. Using noise.")
        return np.random.normal(0, 0.01, int(sample_rate * duration))

# Get file lists
voice_files = [os.path.join(input_voice_dir, f) for f in os.listdir(input_voice_dir) if f.endswith('.flac')]
noise_files = [os.path.join(input_noise_dir, f) for f in os.listdir(input_noise_dir) if f.endswith('.wav')]
random.shuffle(noise_files)  # Randomize noise selection

# Convert voice (100 FLAC samples to WAV)
for i, voice_file in enumerate(voice_files[:num_samples_per_class]):
    convert_and_process_audio(voice_file, f'{output_dir}/pure_voice/sample_{i}.wav', is_flac=True)

# Process noise (100 WAV samples)
for i, noise_file in enumerate(noise_files[:num_samples_per_class]):
    convert_and_process_audio(noise_file, f'{output_dir}/line_noise/sample_{i}.wav', is_flac=False)

# Generate empty (100 samples)
for i in range(num_samples_per_class):
    audio = generate_empty()
    sf.write(f'{output_dir}/empty/sample_{i}.wav', audio, sample_rate)

# Generate noise+voice (100 samples)
for i in range(num_samples_per_class):
    voice_file = f'{output_dir}/pure_voice/sample_{i}.wav'
    noise_file = f'{output_dir}/line_noise/sample_{i}.wav'
    mixed = mix_audio(voice_file, noise_file)
    sf.write(f'{output_dir}/noise_voice/sample_{i}.wav', mixed, sample_rate)

print(f'Converted and organized dataset in {output_dir}')