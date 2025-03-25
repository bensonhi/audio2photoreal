import os
import json
import torch
import whisper
from tqdm import tqdm
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional

def transcribe_audio(
    audio_path: str,
    model_name: str = "base",
    device: Optional[str] = None
) -> Dict:
    """
    Transcribe audio file using Whisper model.
    
    Args:
        audio_path: Path to the audio file
        model_name: Whisper model size (tiny, base, small, medium, large)
        device: Device to run the model on (cuda, cpu)
    
    Returns:
        Dictionary containing transcription and timing information
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = whisper.load_model(model_name).to(device)
    result = model.transcribe(audio_path)
    
    return result

def process_dataset(
    dataset_root: str,
    output_dir: str,
    model_name: str = "base",
    device: Optional[str] = None
) -> None:
    """
    Process all WAV files in the dataset directory and save transcriptions.
    
    Args:
        dataset_root: Root directory containing dataset folders
        output_dir: Directory to save transcriptions
        model_name: Whisper model size
        device: Device to run the model on
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset folder
    for dataset_folder in dataset_root.iterdir():
        if not dataset_folder.is_dir():
            continue
            
        print(f"Processing dataset: {dataset_folder.name}")
        dataset_output_dir = output_dir / dataset_folder.name
        dataset_output_dir.mkdir(exist_ok=True)
        
        # Find all WAV files recursively
        wav_files = list(dataset_folder.rglob("*.wav"))
        
        for wav_file in tqdm(wav_files, desc=f"Transcribing {dataset_folder.name}"):
            try:
                # Transcribe audio
                result = transcribe_audio(str(wav_file), model_name, device)
                
                # Create output filename
                rel_path = wav_file.relative_to(dataset_folder)
                output_file = dataset_output_dir / rel_path.with_suffix('.json')
                
                # Ensure output directory exists
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save transcription
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"Error processing {wav_file}: {str(e)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper")
    parser.add_argument("--dataset_root", type=str, required=True,
                      help="Root directory containing dataset folders")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save transcriptions")
    parser.add_argument("--model_name", type=str, default="base",
                      choices=["tiny", "base", "small", "medium", "large"],
                      help="Whisper model size")
    parser.add_argument("--device", type=str, default=None,
                      help="Device to run the model on (cuda, cpu)")
    
    args = parser.parse_args()
    
    process_dataset(
        args.dataset_root,
        args.output_dir,
        args.model_name,
        args.device
    )

if __name__ == "__main__":
    main() 