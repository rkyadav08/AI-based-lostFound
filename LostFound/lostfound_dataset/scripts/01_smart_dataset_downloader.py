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

class SmartDatasetDownloader:
    def __init__(self):
        self.base_dir = "lostfound_dataset"
        os.makedirs(f"{self.base_dir}/images", exist_ok=True)
        os.makedirs(f"{self.base_dir}/embeddings", exist_ok=True)
        self.metadata = []
        self.failed_downloads = []
        
        # Initialize CLIP for auto-categorization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.categories = [
            "wallet", "phone", "keys", "bag", "jewelry", 
            "electronics", "clothing", "documents", "other"
        ]

    def predict_category(self, description):
        """Use CLIP to predict the most relevant category"""
        text_inputs = clip.tokenize([f"a photo of {desc}" for desc in self.categories]).to(self.device)
        desc_input = clip.tokenize([description]).to(self.device)
        
        with torch.no_grad():
            category_features = self.model.encode_text(text_inputs)
            desc_features = self.model.encode_text(desc_input)
        
        similarities = (desc_features @ category_features.T).softmax(dim=-1)
        return self.categories[similarities.argmax().item()]

    def generate_filename(self, url, description):
        """Generate filename from description and hash"""
        ext = os.path.splitext(urlparse(url).path)[1] or '.jpg'
        clean_desc = ''.join(c if c.isalnum() else '_' for c in description.lower())[:50]
        return f"{clean_desc}_{uuid.uuid4().hex[:6]}{ext}"

    def validate_image(self, img):
        """Quality checks for downloaded images"""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if min(img.size) < 300:
            raise ValueError(f"Resolution too small: {img.size}")
        return img

    def download_single_item(self, url, description):
        """Process one item with automatic categorization"""
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                
                img = Image.open(BytesIO(response.content))
                img = self.validate_image(img)
                
                # Auto-categorize
                category = self.predict_category(description)
                filename = self.generate_filename(url, description)
                save_path = f"{self.base_dir}/images/{filename}"
                
                # Enhanced preprocessing
                img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
                img.save(save_path, quality=85)
                
                return {
                    "id": filename.split('.')[0],
                    "filename": filename,
                    "description": description,
                    "category": category,
                    "source_url": url,
                    "resolution": f"{img.size[0]}x{img.size[1]}",
                    "download_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except Exception as e:
                if attempt == 2:
                    self.failed_downloads.append({
                        "url": url,
                        "error": str(e),
                        "attempts": attempt + 1
                    })
                time.sleep(2 ** attempt)
        return None

    def download_from_source(self, source_file):
        """Process input CSV/JSON with columns: description,url"""
        if source_file.endswith('.csv'):
            items = pd.read_csv(source_file).to_dict('records')
        else:  # Assume JSON
            items = pd.read_json(source_file).to_dict('records')

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.download_single_item, item['url'], item['description']) 
                      for item in items]
            
            for future in concurrent.futures.as_completed(futures):
                if result := future.result():
                    self.metadata.append(result)
                    print(f"Downloaded: {result['filename']} ({result['category']})")

        self.save_metadata()
        self.generate_report()

    def save_metadata(self):
        """Save metadata with auto-generated categories"""
        df = pd.DataFrame(self.metadata)
        df.to_csv(f"{self.base_dir}/metadata.csv", index=False)
        print(f"\nSuccessfully processed {len(self.metadata)} items")
        print("Category distribution:")
        print(df['category'].value_counts())

    def generate_report(self):
        """Generate error report"""
        if self.failed_downloads:
            pd.DataFrame(self.failed_downloads).to_csv(
                f"{self.base_dir}/download_errors.csv", index=False
            )
            print(f"\nFailed downloads: {len(self.failed_downloads)}")

if __name__ == "__main__":
    downloader = SmartDatasetDownloader()
    
    # Example: python 01_smart_dataset_downloader.py input.csv
    import sys
    if len(sys.argv) > 1:
        source_file = sys.argv[1]
    else:
        source_file = "items.csv"  # Default input file
    
    if os.path.exists(source_file):
        downloader.download_from_source(source_file)
    else:
        print(f"Error: Input file {source_file} not found")
        print("Expected CSV/JSON with columns: description,url")