#!/usr/bin/env python3
import os
import argparse
import sys
import time
from pathlib import Path
import openai

# Ensure your OpenAI API key is set in the environment:
# export OPENAI_API_KEY="your_api_key_here"


def transcribe_audio_file(audio_path: Path) -> str:
    """
    Sends the given audio file to OpenAI Whisper (whisper-1) and returns the transcript text.
    """
    with audio_path.open("rb") as audio_file:
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
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Initial seconds to wait between API retries (default: 1.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retries on rate limit errors (default: 5)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining files if a transcription fails",
    )
    args = parser.parse_args()

    # Determine source path (file or folder)
    if args.input:
        source = Path(args.input).expanduser()
    else:
        source = (
            Path.home()
            / "Library/Group Containers/243LU875E5.groups.com.apple.podcasts/Library/Cache"
        )

    if not source.exists():
        print(f"Error: path '{source}' does not exist.")
        sys.exit(1)

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
        sys.exit(0)

    # Process files
    total = len(audio_files)
    for idx, audio_path in enumerate(audio_files, 1):
        print(f"[{idx}/{total}] Transcribing: {audio_path}")
        retries = args.max_retries
        backoff = args.delay
        transcript = None

        while retries > 0:
            try:
                transcript = transcribe_audio_file(audio_path)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    print(
                        f"Rate limit hit. Retrying in {backoff:.1f}s... ({retries-1} retries left)"
                    )
                    time.sleep(backoff)
                    backoff *= 2  # exponential backoff
                    retries -= 1
                    continue
                else:
                    print(f"Error processing {audio_path.name}: {e}")
                    if not args.continue_on_error:
                        sys.exit(1)
                    break

        if transcript is None:
            print(
                f"Failed to transcribe {audio_path.name} after {args.max_retries} retries."
            )
            if not args.continue_on_error:
                sys.exit(1)
            else:
                continue

        # Save transcript
        out_path = output_dir / f"{audio_path.stem}.txt"
        out_path.write_text(transcript, encoding="utf-8")
        print(f"Saved: {out_path}")

    print("All done.")


if __name__ == "__main__":
    main()
