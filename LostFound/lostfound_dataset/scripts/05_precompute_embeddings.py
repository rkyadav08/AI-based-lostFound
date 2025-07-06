# scripts/05_precompute_embeddings.py
import os
import torch
import clip
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

class EmbeddingGenerator:
    def __init__(self, base_dir: str = "lostfound_dataset"):
        self.base_dir = base_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize CLIP model (reusing your existing setup)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        
        # Optimizations
        if self.device == "cuda":
            self.model = self.model.half()  # FP16 for faster inference
            torch.backends.cudnn.benchmark = True

    def load_metadata(self) -> pd.DataFrame:
        """Load and validate metadata"""
        metadata_path = os.path.join(self.base_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        df = pd.read_csv(metadata_path)
        required_cols = {"filename", "user_description", "ai_description"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Metadata missing required columns: {required_cols}")
            
        return df

    def process_single_image(self, row: Dict) -> Dict:
        """Process one image and return embeddings"""
        try:
            img_path = os.path.join(self.base_dir, "images", row["filename"])
            image = Image.open(img_path)
            
            # Apply your existing preprocessing
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
            
            # CLIP processing
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if self.device == "cuda":
                    image_input = image_input.half()  # FP16 if GPU available
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            return {
                "filename": row["filename"],
                "embedding": image_features.cpu().numpy().astype('float32')[0],
                "status": "success"
            }
        except Exception as e:
            return {
                "filename": row.get("filename", "unknown"),
                "error": str(e),
                "status": "failed"
            }

    def generate_embeddings(self, batch_size: int = 16) -> None:
        """Main processing pipeline"""
        # Load metadata
        df = self.load_metadata()
        print(f"Processing {len(df)} images...")
        
        # Initialize FAISS index
        dimension = 512  # CLIP feature dimension
        index = faiss.IndexFlatL2(dimension)
        
        # Process in parallel batches
        successful = []
        failed = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _, row in df.iterrows():
                futures.append(executor.submit(self.process_single_image, row))
                
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(df)):
                result = future.result()
                if result["status"] == "success":
                    successful.append(result)
                else:
                    failed.append(result)
        
        # Handle results
        if not successful:
            raise RuntimeError("All image processing failed!")
            
        # Create embeddings matrix
        embeddings = np.array([x["embedding"] for x in successful]).astype('float32')
        filenames = [x["filename"] for x in successful]
        
        # Add to FAISS index
        index.add(embeddings)
        
        # Save outputs
        self.save_outputs(index, filenames, embeddings, failed)

    def save_outputs(self, index, filenames, embeddings, failed):
        """Save all generated artifacts"""
        # Save FAISS index
        faiss.write_index(index, os.path.join(self.base_dir, "embeddings", "clip_embeddings.index"))
        
        # Save embeddings as numpy array
        np.save(os.path.join(self.base_dir, "embeddings", "clip_embeddings.npy"), embeddings)
        
        # Save filename mapping
        pd.DataFrame({
            "filename": filenames,
            "embedding_index": range(len(filenames))
        }).to_csv(os.path.join(self.base_dir, "embeddings", "filename_mapping.csv"), index=False)
        
        # Save error log if any
        if failed:
            pd.DataFrame(failed).to_csv(
                os.path.join(self.base_dir, "embeddings", "processing_errors.csv"),
                index=False
            )
        
        print(f"\nSuccessfully processed {len(filenames)} images")
        if failed:
            print(f"Failed on {len(failed)} images (see processing_errors.csv)")

if __name__ == "__main__":
    print("=== CLIP Embedding Generator ===")
    generator = EmbeddingGenerator()
    
    try:
        generator.generate_embeddings()
        print("\nEmbedding generation complete!")
        print("Output files:")
        print(f"- clip_embeddings.index (FAISS index)")
        print(f"- clip_embeddings.npy (Raw embeddings)")
        print(f"- filename_mapping.csv (Index to filename mapping)")
    except Exception as e:
        print(f"\nError: {str(e)}")