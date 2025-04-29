#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import openai

# Ensure your OpenAI API key is set in the environment:
# export OPENAI_API_KEY="your_api_key_here"


def transcribe_audio_file(audio_path: Path) -> str:
    """
    Sends the given audio file to OpenAI Whisper (whisper-1) and returns the transcript text.
    """
    with audio_path.open("rb") as audio_file:
        # For openai-python >=1.0.0, use the new audio.transcriptions endpoint
        response = openai.audio.transcriptions.create(
            file=audio_file, model="whisper-1"
        )
    return response.get("text", "")


def find_audio_files(base_dir: Path) -> list[Path]:
    """
    Recursively find all .mp3 files under base_dir.
    """
    return list(base_dir.rglob("*.mp3"))


def main():
    parser = argparse.ArgumentParser(
        description="Batch-transcribe .mp3 files to text using OpenAI Whisper"
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to an .mp3 file or directory to search for .mp3 files.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="transcripts",
        help="Directory in which to save .txt transcripts (default: ./transcripts)",
    )
    args = parser.parse_args()

    # Determine source path (file or folder)
    if args.input:
        source = Path(args.input).expanduser()
    else:
        # Default podcasts cache path on macOS
        source = (
            Path.home()
            / "Library/Group Containers/243LU875E5.groups.com.apple.podcasts/Library/Cache"
        )

    if not source.exists():
        print(f"Error: path '{source}' does not exist.")
        return

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect audio files
    if source.is_file() and source.suffix.lower() == ".mp3":
        audio_files = [source]
    else:
        audio_files = find_audio_files(source)

    if not audio_files:
        print("No .mp3 files found.")
        return

    # Transcribe each file
    for audio_path in audio_files:
        print(f"Transcribing: {audio_path}")
        try:
            transcript = transcribe_audio_file(audio_path)
            out_path = output_dir / f"{audio_path.stem}.txt"
            out_path.write_text(transcript, encoding="utf-8")
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Failed {audio_path}: {e}")


if __name__ == "__main__":
    main()
