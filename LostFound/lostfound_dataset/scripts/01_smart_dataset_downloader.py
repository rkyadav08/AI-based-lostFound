# scripts/01_smart_dataset_downloader.py
import os
import requests
import uuid
from PIL import Image, ImageFilter
from io import BytesIO
import pandas as pd
import time
from urllib.parse import urlparse
import concurrent.futures
import clip
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from typing import Optional, Dict, List
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

class EnhancedLostItemSearch:
    def __init__(self):
        """Initialize with paths and enhanced AI models"""
        self.base_dir = os.path.join(os.path.dirname(__file__), "..")
        os.makedirs(os.path.join(self.base_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "embeddings"), exist_ok=True)
        self.metadata = []
        self.failed_downloads = []
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize CLIP for visual embeddings
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        
        # Initialize BLIP for detailed captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(self.device)
        
        # Initialize object detection pipeline
        self.detector = pipeline(
            "object-detection", 
            model="facebook/detr-resnet-50",
            device=0 if self.device == "cuda" else -1
        )
        
        # Initialize color analysis tools
        self.color_names = {
            (255,0,0): 'red', (0,255,0): 'green', (0,0,255): 'blue',
            (255,255,0): 'yellow', (255,0,255): 'magenta', (0,255,255): 'cyan',
            (128,0,0): 'dark red', (0,128,0): 'dark green', (0,0,128): 'dark blue',
            (128,128,0): 'olive', (128,0,128): 'purple', (0,128,128): 'teal',
            (192,192,192): 'silver', (128,128,128): 'gray', (255,255,255): 'white',
            (0,0,0): 'black', (255,165,0): 'orange', (255,192,203): 'pink'
        }

    def validate_url(self, url: str) -> bool:
        """Enhanced URL validation with content type checking"""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False
                
            response = requests.head(url, timeout=5, allow_redirects=True)
            content_type = response.headers.get('content-type', '')
            return (response.status_code == 200 and 
                    any(x in content_type for x in ['image/jpeg', 'image/png', 'image/webp']))
        except:
            return False

    def download_image(self, url: str) -> Image.Image:
        """Download and validate an image with enhanced checks"""
        try:
            response = requests.get(url, timeout=15, stream=True)
            response.raise_for_status()
            
            # Verify image content
            img = Image.open(BytesIO(response.content))
            img.verify()
            img = Image.open(BytesIO(response.content))  # Reopen after verify
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Basic quality check
            if img.size[0] < 50 or img.size[1] < 50:
                raise ValueError("Image resolution too small")
                
            return img
        except Exception as e:
            raise ValueError(f"Image download failed: {str(e)}")

    def generate_filename(self, description: str) -> str:
        """Create a clean filename with item category"""
        # Extract potential category from description
        categories = ['wallet', 'phone', 'keys', 'bag', 'jewelry', 'clothing', 'electronics', 'documents', 'other']
        category = next((cat for cat in categories if cat in description.lower()), 'other')
        
        clean_desc = ''.join(c if c.isalnum() else '_' for c in description.lower())[:30]
        return f"{category}_{clean_desc}_{uuid.uuid4().hex[:6]}.jpg"

    def get_dominant_colors(self, image: Image.Image, k=3) -> List[str]:
        """Extract dominant colors from image"""
        # Resize for faster processing
        img = image.resize((100, 100))
        img_array = np.array(img)
        
        # Reshape to list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Find dominant colors
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pixels)
        
        # Get color labels and counts
        counts = Counter(kmeans.labels_)
        sorted_colors = [kmeans.cluster_centers_[i] for i, _ in counts.most_common()]
        
        # Map to color names
        color_names = []
        for color in sorted_colors:
            # Find closest named color
            min_dist = float('inf')
            closest_color = None
            for rgb, name in self.color_names.items():
                dist = sum((c1 - c2)**2 for c1, c2 in zip(color, rgb))
                if dist < min_dist:
                    min_dist = dist
                    closest_color = name
            color_names.append(closest_color)
        
        return color_names

    def detect_objects(self, image: Image.Image) -> List[str]:
        """Detect prominent objects in the image"""
        try:
            results = self.detector(image)
            return [obj['label'] for obj in results if obj['score'] > 0.7]
        except:
            return []

    def generate_detailed_description(self, image_path: str) -> Dict:
        """Generate comprehensive description with multiple aspects"""
        try:
            raw_image = Image.open(image_path).convert('RGB')
            
            # Generate base caption
            inputs = self.blip_processor(raw_image, return_tensors="pt").to(self.device)
            generated_ids = self.blip_model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=7,
                temperature=0.9
            )
            base_caption = self.blip_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            # Get additional details
            dominant_colors = self.get_dominant_colors(raw_image)
            detected_objects = self.detect_objects(raw_image)
            
            # Generate structured description
            description = {
                "base_description": base_caption,
                "colors": dominant_colors,
                "materials": self.detect_materials(base_caption),
                "objects": detected_objects,
                "size_estimate": self.estimate_size(raw_image),
                "text_present": self.detect_text(raw_image),
                "brand_indicators": self.detect_brand_indicators(base_caption)
            }
            
            return description
            
        except Exception as e:
            print(f"Detailed description failed: {str(e)}")
            return None

    def detect_materials(self, caption: str) -> List[str]:
        """Extract materials from caption"""
        materials = ['leather', 'metal', 'plastic', 'fabric', 'wood', 'glass', 'paper', 'ceramic']
        return [mat for mat in materials if mat in caption.lower()]

    def estimate_size(self, image: Image.Image) -> str:
        """Estimate relative size of object"""
        area = image.size[0] * image.size[1]
        if area > 1000000:
            return "large"
        elif area > 100000:
            return "medium"
        else:
            return "small"

    def detect_text(self, image: Image.Image) -> bool:
        """Check if image contains text elements"""
        # Simple heuristic - more sophisticated OCR could be added
        edges = image.filter(ImageFilter.FIND_EDGES)
        edge_pixels = np.array(edges).mean()
        return edge_pixels > 30  # Threshold for text-like edges

    def detect_brand_indicators(self, caption: str) -> List[str]:
        """Extract potential brand indicators"""
        brand_keywords = ['logo', 'brand', 'label', 'tag', 'inscription']
        return [kw for kw in brand_keywords if kw in caption.lower()]

    def generate_clip_embedding(self, image_path: str) -> np.ndarray:
        """Generate CLIP embedding for visual search"""
        try:
            image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image)
            return embedding.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"CLIP embedding failed: {str(e)}")
            return None

    def process_item(self, url: str, description: str) -> Dict:
        """Enhanced item processing with rich metadata"""
        try:
            # Validate and download
            if not self.validate_url(url):
                raise ValueError("Invalid URL or inaccessible")
            
            img = self.download_image(url)
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
            
            # Save image
            filename = self.generate_filename(description)
            save_path = os.path.join(self.base_dir, "images", filename)
            img.save(save_path, quality=85)
            
            # Generate rich descriptions
            detailed_description = self.generate_detailed_description(save_path)
            clip_embedding = self.generate_clip_embedding(save_path)
            
            # Save embedding if generated
            if clip_embedding is not None:
                embedding_path = os.path.join(self.base_dir, "embeddings", f"{os.path.splitext(filename)[0]}.npy")
                np.save(embedding_path, clip_embedding)
            
            return {
                "filename": filename,
                "user_description": description,
                "ai_description": detailed_description,
                "url": url,
                "resolution": f"{img.width}x{img.height}",
                "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "embedding_path": embedding_path if clip_embedding is not None else None,
                "category": filename.split('_')[0]  # Extract category from filename
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "url": url,
                "description": description
            }

    def run(self):
        """Main execution flow with enhanced reporting"""
        input_file = os.path.join(os.path.dirname(__file__), "items.csv")
        
        try:
            # Load input data
            items = pd.read_csv(input_file)
            if len(items) == 0:
                raise ValueError("CSV file is empty")
                
            print(f"Found {len(items)} items to process...")
            print("Generating rich descriptions for better searchability...")
            
            # Process items with thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for _, row in items.iterrows():
                    if pd.isna(row['url']) or pd.isna(row['description']):
                        continue
                    futures.append(executor.submit(
                        self.process_item,
                        row['url'].strip(),
                        str(row['description']).strip()
                    ))
                
                # Collect results with progress feedback
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    result = future.result()
                    if 'error' in result:
                        self.failed_downloads.append(result)
                        print(f"❌ Failed: {result['description']} - {result['error']}")
                    else:
                        self.metadata.append(result)
                        print(f"✅ Processed {i+1}/{len(items)}: {result['filename']}")
                        if result['ai_description']:
                            print(f"   Colors: {', '.join(result['ai_description']['colors'])}")
                            print(f"   Objects: {', '.join(result['ai_description']['objects'])}")
            
            # Save outputs
            self.save_results()
            
        except Exception as e:
            print(f"\nFATAL ERROR: {str(e)}")
            print("\nTroubleshooting:")
            print(f"1. Verify {input_file} exists and contains 'description,url' columns")
            print("2. Check your internet connection")
            print("3. Ensure URLs point directly to images (not web pages)")

    def save_results(self):
        """Save all outputs with enhanced reporting"""
        # Save successful downloads
        if self.metadata:
            output_path = os.path.join(self.base_dir, "metadata.csv")
            pd.DataFrame(self.metadata).to_csv(output_path, index=False)
            print(f"\nSuccessfully processed {len(self.metadata)} items")
            print(f"Metadata saved to {output_path}")
        
        # Save error log
        if self.failed_downloads:
            error_path = os.path.join(self.base_dir, "download_errors.csv")
            pd.DataFrame(self.failed_downloads).to_csv(error_path, index=False)
            print(f"Failed downloads: {len(self.failed_downloads)}")
            
            # Show top error reasons
            errors = pd.Series([d['error'] for d in self.failed_downloads])
            print("\nMost common errors:")
            print(errors.value_counts().head(5))

if __name__ == "__main__":
    print("=== Enhanced Lost & Found Search System ===")
    print("Initializing AI models... (this may take a moment)")
    
    processor = EnhancedLostItemSearch()
    processor.run()
    
    print("\nProcessing complete. The system has generated:")
    print("- High-quality images of lost items")
    print("- Detailed descriptions with colors, materials, and objects")
    print("- Visual embeddings for similarity search")
    print("- Categorized metadata for efficient searching")
