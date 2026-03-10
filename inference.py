"""XTTS emotion inference helper.

This CLI script loads an XTTS checkpoint that was finetuned with
emotion embeddings and synthesises speech from the provided text.

Usage:
    python inference.py \
        --config outputs/run/config.json \
        --checkpoint outputs/run/best_model.pth \
        --tokenizer outputs/XTTS_v2.0_original_model_files/vocab.json \
        --speaker-wav emotion-data/speaker_1/angry_000276.wav \
        --text "Bugün harika hissediyorum" \
        --emotion happy \
        --output out.wav

Multiple --speaker-wav arguments can be supplied to average reference
embeddings. Extra decoding parameters (temperature, top-p, top-k, etc.)
are exposed as optional flags.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List

import torch
import torchaudio

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XTTS emotion inference")
    parser.add_argument("--config", type=Path, required=True, help="Path to the exported XTTS config.json")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the XTTS checkpoint (.pth)")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to the tokenizer vocab.json")
    parser.add_argument(
        "--speaker-wav",
        type=Path,
        required=True,
        action="append",
        help="Reference WAV(s) for conditioning. Provide multiple times to average embeddings.",
    )
    parser.add_argument("--text", type=str, required=True, help="Input text to synthesise")
    parser.add_argument("--language", type=str, default="tr", help="Language code used during finetuning")
    parser.add_argument(
        "--emotion",
        type=str,
        default="neutral",
        help="Emotion label to condition on (e.g. neutral, angry, happy).",
    )
    parser.add_argument(
        "--emotion-id",
        type=int,
        default=None,
        help="Override emotion label with a numeric id (takes precedence over --emotion).",
    )
    parser.add_argument("--output", type=Path, default=Path("xtts_inference.wav"), help="Output WAV file path")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda, cpu). Defaults to auto")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.85, help="Top-p nucleus sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=10.0, help="Repetition penalty")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="Length penalty")
    parser.add_argument("--gpt-cond-len", type=int, default=6, help="Reference length in seconds for GPT")
    parser.add_argument(
        "--gpt-cond-chunk-len",
        type=int,
        default=6,
        help="Chunk length in seconds for GPT conditioning (<= gpt-cond-len)",
    )
    parser.add_argument("--max-ref-len", type=int, default=10, help="Max seconds of reference audio to use")
    parser.add_argument(
        "--normalize-refs",
        action="store_true",
        help="Apply amplitude normalization to reference audio before embedding.",
    )
    return parser.parse_args(list(argv))


def ensure_paths(paths: Iterable[Path]) -> None:
    missing: List[str] = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
    if missing:
        formatted = "\n".join(missing)
        raise FileNotFoundError(f"The following paths do not exist:\n{formatted}")


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(args: argparse.Namespace, device: torch.device) -> Xtts:
    config = XttsConfig()
    config.load_json(str(args.config))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(args.checkpoint),
        vocab_path=str(args.tokenizer),
        use_deepspeed=False,
    )
    model.to(device)
    return model


def compute_conditioning_latents(model: Xtts, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    wav_paths = [str(path) for path in args.speaker_wav]
    return model.get_conditioning_latents(
        audio_path=wav_paths,
        gpt_cond_len=args.gpt_cond_len,
        gpt_cond_chunk_len=args.gpt_cond_chunk_len,
        max_ref_length=args.max_ref_len,
        sound_norm_refs=args.normalize_refs,
    )


def synthesize(model: Xtts, args: argparse.Namespace, gpt_cond_latent, speaker_embedding) -> dict:
    start_time = time.time()
    outputs = model.inference(
        text=args.text,
        language=args.language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        do_sample=True,
        num_beams=1,
        emotion=args.emotion,
        emotion_id=args.emotion_id,
    )
    elapsed = time.time() - start_time
    print(f"Inference completed in {elapsed:.2f}s")
    return outputs


def save_audio(output_path: Path, audio, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(torch.float32)
    torchaudio.save(str(output_path), audio_tensor, sample_rate)
    print(f"Saved waveform to {output_path}")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    ensure_paths([args.config, args.checkpoint, args.tokenizer, *args.speaker_wav])
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = load_model(args, device)
    model.eval()

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = compute_conditioning_latents(model, args)
    gpt_cond_latent = gpt_cond_latent.to(device)
    speaker_embedding = speaker_embedding.to(device)

    print("Synthesising...")
    outputs = synthesize(model, args, gpt_cond_latent, speaker_embedding)

    wav = outputs.get("wav")
    if wav is None:
        raise RuntimeError("Model inference did not return a 'wav' entry")
    save_audio(args.output, wav, sample_rate=model.args.output_sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
