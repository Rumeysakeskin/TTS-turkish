import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Add here the xtts_config path
path = "/root/TTS-turkish/outputs/"

CONFIG_PATH = path + "XTTS-Emotion-TR-September-27-2025_07+39PM-f0b76c07/config.json"
# Add here the vocab file that you have used to train the model
TOKENIZER_PATH = path + "XTTS_v2.0_original_model_files/vocab.json"
# Add here the checkpoint that you want to do inference with
XTTS_CHECKPOINT = path + "XTTS-Emotion-TR-September-27-2025_07+39PM-f0b76c07/best_model.pth"
# Add here the speaker reference
SPEAKER_REFERENCE = path + "angry_000000.wav"

# Function to check if file exists
def check_file(path):
    if os.path.exists(path):
        print(f"File exists: {path}")
    else:
        print(f"File does not exist: {path}")

print("Loading model...")
config = XttsConfig()
# Check each file
check_file(CONFIG_PATH)
check_file(TOKENIZER_PATH)
check_file(XTTS_CHECKPOINT)
check_file(SPEAKER_REFERENCE)

config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)

model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

print("Inference...")
import time
start = time.time()
out = model.inference(
    "Buharlı temizleme programında sıcaklık ayarı yapılamıyor bu program için varsayılan sıcaklık değeri yüz derecedir",
    "tr",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
print("inference time:", time.time()- start)

# output wav path
OUTPUT_WAV_PATH = path + "mtts-angry-inference.wav"

# Function to save audio
def save_audio(output_path, audio_data, sample_rate):
    try:
        # Extract directory path from the output path
        output_dir = os.path.dirname(output_path)
        
        # Check if the directory exists, if not create it
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory created: {output_dir}")

        # Save the audio file
        torchaudio.save(output_path, audio_data, sample_rate)
        print(f"Audio saved successfully at {output_path}")
    except Exception as e:
        print(f"Failed to save audio: {e}")

# Assuming 'out' is a dictionary containing 'wav' key with audio data
audio_tensor = torch.tensor(out["wav"]).unsqueeze(0)
sample_rate = 24000

# Save the audio file
save_audio(OUTPUT_WAV_PATH, audio_tensor, sample_rate)