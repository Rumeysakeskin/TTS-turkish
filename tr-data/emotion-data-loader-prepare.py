import os
import re
import unicodedata
import subprocess
from datasets import load_dataset
import soundfile as sf
from collections import defaultdict

# --- NORMALIZATION HELPERS ---
NUMBERS = {0: "sıfır", 1: "bir", 2: "iki", 3: "üç", 4: "dört", 5: "beş", 6: "altı",
           7: "yedi", 8: "sekiz", 9: "dokuz", 10: "on", 20: "yirmi", 30: "otuz",
           40: "kırk", 50: "elli", 60: "altmış", 70: "yetmiş", 80: "seksen", 90: "doksan"}

rep_map = {",": " ", ".": " ", ";": " ", ":": " ", "!": " ", "?": " ",
           "\"": " ", "“": " ", "”": " ", "’": " ", "'": " ", "…": " ", "-": " "}

def number_to_turkish(num):
    if num == 0: return "sıfır"
    if num < 0: return f"eksi {number_to_turkish(-num)}"
    if num in NUMBERS: return NUMBERS[num]
    if num < 100:
        tens = (num // 10) * 10
        ones = num % 10
        return f"{NUMBERS[tens]} {NUMBERS[ones]}".strip()
    elif num < 1000:
        hundreds = num // 100
        remainder = num % 100
        hundreds_str = "yüz" if hundreds == 1 else f"{NUMBERS[hundreds]} yüz"
        return hundreds_str if remainder == 0 else f"{hundreds_str} {number_to_turkish(remainder)}"
    elif num < 1000000:
        thousands = num // 1000
        remainder = num % 1000
        thousands_str = "bin" if thousands == 1 else f"{number_to_turkish(thousands)} bin"
        return thousands_str if remainder == 0 else f"{thousands_str} {number_to_turkish(remainder)}"
    elif num < 10000000:
        millions = num // 1000000
        remainder = num % 1000000
        millions_str = "bir milyon" if millions == 1 else f"{number_to_turkish(millions)} milyon"
        return millions_str if remainder == 0 else f"{millions_str} {number_to_turkish(remainder)}"
    else:
        return str(num)

def normalize_abbreviations(text):
    text = re.sub(r'\bTL\b(?!\')', 'türk lirası', text, flags=re.IGNORECASE)
    text = re.sub(r'\bABD\b(?!\')', 'amerika birleşik devletleri', text, flags=re.IGNORECASE)
    text = re.sub(r'\bM\.?\s*Ö\.?\b', 'milattan önce', text, flags=re.IGNORECASE)
    return text

def normalize_symbols(text):
    text = re.sub(r'%', ' yüzde', text)
    text = re.sub(r'\+', ' artı ', text)
    return text

def normalize_numbers(text):
    def replace_number(match):
        num = int(match.group())
        if 0 <= num <= 9999999:
            return number_to_turkish(num)
        return match.group()
    return re.sub(r'\b\d{1,7}\b', replace_number, text)

def normalize_text(text):
    if not text or not isinstance(text, str):
        return text

    # Köşeli parantez içindekileri kaldır
    text = re.sub(r"\[.*?\]", " ", text)

    text = unicodedata.normalize('NFC', text)
    text = normalize_abbreviations(text)
    text = normalize_numbers(text)
    text = normalize_symbols(text)
    text = text.lower()
    for old, new in rep_map.items():
        text = text.replace(old, new)
    return re.sub(r'\s+', ' ', text).strip()


# --- WAV CHECK/CONVERT ---
def ensure_wav_format(wav_path):
    """22050 Hz, mono, 16-bit değilse sox ile düzelt."""
    try:
        info = subprocess.check_output(["sox", "--i", wav_path], text=True)
        sr = int(re.search(r"Sample Rate\s+:\s+(\d+)", info).group(1))
        ch = int(re.search(r"Channels\s+:\s+(\d+)", info).group(1))
        enc = re.search(r"Precision\s+:\s+(\d+)-bit", info).group(1)

        if sr == 22050 and ch == 1 and enc == "16":
            return  # zaten uygun

        print(f"⚠️ Dönüştürülüyor: {wav_path} ({sr}Hz, {ch}ch, {enc}-bit)")
        tmp_path = wav_path + ".tmp.wav"
        subprocess.run([
            "sox", wav_path, "-r", "22050", "-c", "1", "-b", "16", tmp_path
        ], check=True)
        os.replace(tmp_path, wav_path)

    except Exception as e:
        print(f"sox kontrol hatası: {e}")

# --- DATASET LOAD ---
dataset = load_dataset("Codyfederer/tr-full-dataset", split="train")

# Script'in bulunduğu dizin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# emotion-data klasörü parent dizine oluşturulsun
BASE_DIR = os.path.join(parent_dir, "emotion-data")
os.makedirs(BASE_DIR, exist_ok=True)

print("Oluşturulacak klasör:", BASE_DIR)

# --- SPEAKER SÜRE SAYACI ---
speaker_durations = defaultdict(float)  # saniye cinsinden
emotion_counts = defaultdict(int)  # emotion sayıları
total_duration = 0.0

metadata_path = os.path.join(BASE_DIR, "metadata.txt")
with open(metadata_path, "w", encoding="utf-8") as out_f:
    # XTTS Emotion Dataset - metadata.txt header
    out_f.write("# XTTS Emotion Dataset - metadata.txt\n")
    out_f.write("# Format: wav_file_path|text|speaker_id|emotion\n")
    out_f.write("# Supported emotions: neutral, angry, sad, happy\n\n")
    for i, sample in enumerate(dataset):
        text = normalize_text(sample["text"])
        speaker = str(sample["speaker_id"])
        # Dataset'te sadece 4 emotion var: neutral, angry, sad, happy
        emotion = str(sample.get("emotion", "neutral")).lower()
        
        # Sadece desteklenen emotion'ları kabul et
        supported_emotions = {"neutral", "angry", "sad", "happy"}
        if emotion not in supported_emotions:
            emotion = "neutral"  # Bilinmeyen emotion'lar için neutral
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]

        # Süre hesapla
        duration = len(audio) / sr
        speaker_durations[speaker] += duration
        emotion_counts[emotion] += 1
        total_duration += duration

        # XTTS için daha düzenli klasör yapısı
        speaker_dir = os.path.join(BASE_DIR, f"speaker_{speaker}")
        os.makedirs(speaker_dir, exist_ok=True)

        # Dosya adını emotion ile birlikte oluştur
        wav_name = f"{emotion}_{i:06d}.wav"
        lab_name = f"{emotion}_{i:06d}.lab"
        wav_path = os.path.join(speaker_dir, wav_name)
        lab_path = os.path.join(speaker_dir, lab_name)

        sf.write(wav_path, audio, sr)
        ensure_wav_format(wav_path)

        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(text)

        # XTTS format: wav_file_path|text|speaker_id|emotion
        # wav_path'i dataset klasörüne göre relative path yap
        relative_wav_path = os.path.relpath(wav_path, BASE_DIR)
        out_f.write(f"{relative_wav_path}|{text}|{speaker}|{emotion}\n")

print(f"\n✅ metadata.txt oluşturuldu: {metadata_path}")

# --- SONUÇLAR ---
def sec_to_hm(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h} saat {m} dk"

print("\n🔊 Speaker Bazlı Süreler:")
for spk, dur in sorted(speaker_durations.items(), key=lambda x: float(x[0])):
    print(f"Speaker {spk}: {sec_to_hm(dur)}")

print(f"\n📊 Toplam Süre: {sec_to_hm(total_duration)}")

print("\n😊 Emotion Dağılımı:")
for emotion in ["neutral", "angry", "sad", "happy"]:
    count = emotion_counts.get(emotion, 0)
    percentage = (count / sum(emotion_counts.values())) * 100 if sum(emotion_counts.values()) > 0 else 0
    print(f"   {emotion}: {count} örnek ({percentage:.1f}%)")
