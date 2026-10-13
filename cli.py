"""Command-line interface for the F5-TTS voice clone service."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import soundfile as sf

from src.engine import VoiceCloneError, VoiceCloneEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clone a voice and synthesize speech with F5-TTS")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--voice", default="deep_male", help="Voice profile name")
    parser.add_argument("--speed", type=float, default=0.9, help="Speech speed (default: 0.9)")
    parser.add_argument("--nfe-step", type=int, default=32, help="Flow-matching steps (default: 32)")
    parser.add_argument("--output", type=Path, help="Optional destination WAV path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.output is not None and args.output.suffix.lower() != ".wav":
        parser.error("--output must use the .wav extension")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        generated_path = Path(
            VoiceCloneEngine().synthesize(
                text=args.text,
                voice_name=args.voice,
                speed=args.speed,
                nfe_step=args.nfe_step,
            )
        )

        final_path = generated_path
        if args.output is not None:
            final_path = args.output.expanduser().resolve()
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path != generated_path.resolve():
                shutil.move(str(generated_path), final_path)

        duration = sf.info(final_path).duration
        print(f"Generated: {final_path}")
        print(f"Duration: {duration:.3f} seconds")
        return 0
    except (VoiceCloneError, ValueError, OSError) as exc:
        logging.getLogger(__name__).error("Synthesis failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
