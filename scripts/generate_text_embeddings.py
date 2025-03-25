import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from google import genai
from typing import Dict, Optional

def generate_embedding(
    text: str,
    client: genai.Client,
    model_name: str = "gemini-embedding-exp-03-07"
) -> Dict:
    """
    Generate embedding for a text using Gemini model.
    
    Args:
        text: Text to generate embedding for
        client: Gemini client
        model_name: Gemini embedding model name
    
    Returns:
        Dictionary containing the generated embedding
    """
    result = client.models.embed_content(
        model=model_name,
        contents=text
    )
    
    return result

def process_transcriptions(
    transcriptions_dir: str,
    output_dir: str,
    api_key: str,
    model_name: str = "gemini-embedding-exp-03-07"
) -> None:
    """
    Process all transcription JSON files and generate embeddings.
    
    Args:
        transcriptions_dir: Directory containing transcription JSON files
        output_dir: Directory to save embeddings
        api_key: Gemini API key
        model_name: Gemini embedding model name
    """
    transcriptions_dir = Path(transcriptions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
    # Process each dataset folder
    for dataset_folder in transcriptions_dir.iterdir():
        if not dataset_folder.is_dir():
            continue
            
        print(f"Processing dataset: {dataset_folder.name}")
        dataset_output_dir = output_dir / dataset_folder.name
        dataset_output_dir.mkdir(exist_ok=True)
        
        # Find all JSON files recursively
        json_files = list(dataset_folder.rglob("*.json"))
        
        for json_file in tqdm(json_files, desc=f"Generating embeddings for {dataset_folder.name}"):
            try:
                # Load transcription
                with open(json_file, 'r', encoding='utf-8') as f:
                    transcription = json.load(f)
                
                # Extract text from transcription
                text = transcription["text"]
                
                # Generate embedding
                embedding_result = generate_embedding(text, client, model_name)
                
                # Create output filename
                rel_path = json_file.relative_to(dataset_folder)
                output_file = dataset_output_dir / rel_path.with_suffix('.embedding.json')
                
                # Ensure output directory exists
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Store embedding along with original text
                print(embedding_result)
                embedding_data = {
                    "text": text,
                    "embedding": embedding_result.embeddings[0].to_dict(),
                    "original_transcription_path": str(json_file)
                }
                
                # Save embedding
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(embedding_data, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for transcriptions using Gemini")
    parser.add_argument("--transcriptions_dir", type=str, required=True,
                      help="Directory containing transcription JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save embeddings")
    parser.add_argument("--api_key", type=str, required=True,
                      help="Gemini API key")
    parser.add_argument("--model_name", type=str, default="gemini-embedding-exp-03-07",
                      help="Gemini embedding model name")
    
    args = parser.parse_args()
    
    process_transcriptions(
        args.transcriptions_dir,
        args.output_dir,
        args.api_key,
        args.model_name
    )

if __name__ == "__main__":
    main() 