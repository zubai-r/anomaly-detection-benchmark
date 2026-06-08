import os
import zipfile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# List of all 15 categories in MVTec AD
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", 
    "hazelnut", "leather", "metal_nut", "pill", "screw", 
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

def download_and_extract_dataset(data_dir="data", category=None):
    """
    Downloads the MVTec AD dataset from Hugging Face Hub (hdtech/mvtech_anomaly_detection)
    and extracts it to the specified data directory.
    """
    target_dir = os.path.join(data_dir, "mvtec_anomaly_detection")
    
    # Check if already extracted and contains all categories (or the requested category)
    if os.path.exists(target_dir):
        nested_dir = os.path.join(target_dir, "mvtech_anomaly_detection")
        check_dir = nested_dir if os.path.exists(nested_dir) else target_dir
        existing_cats = [d for d in os.listdir(check_dir) if os.path.isdir(os.path.join(check_dir, d))]
        if (category is not None and category in existing_cats) or len(existing_cats) >= 15:
            print(f"MVTec AD dataset is already extracted and verified (found category '{category}').")
            return target_dir
            
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "mvtech_anomaly_detection.zip")
    
    if not os.path.exists(zip_path):
        print("Downloading MVTec AD dataset (~5.2 GB) from Hugging Face Hub...")
        # Download the zip file from Hugging Face Hub
        hf_hub_download(
            repo_id="hdtech/mvtech_anomaly_detection",
            filename="mvtech_anomaly_detection.zip",
            repo_type="dataset",
            local_dir=data_dir,
            local_dir_use_symlinks=False
        )
        print("Download complete.")
        
    print(f"Extracting {zip_path} to {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction complete.")
    
    # Clean up the zip file to save space
    try:
        os.remove(zip_path)
        print(f"Removed zip file {zip_path} to save storage space.")
    except Exception as e:
        print(f"Could not remove zip file: {e}")
        
    return target_dir

class MVTecDataset(Dataset):
    def __init__(self, root_path, category, split="train", transform=None, mask_transform=None):
        """
        PyTorch Dataset for loading MVTec AD data.
        Args:
            root_path (str): Path to the extracted mvtec_anomaly_detection folder.
            category (str): Name of the category (e.g. 'bottle').
            split (str): 'train' or 'test'.
            transform (callable): Image transformation.
            mask_transform (callable): Mask transformation.
        """
        self.root_path = root_path
        self.category = category
        self.split = split
        
        # MVTec AD structure: <root>/<category>/train/good/
        # and <root>/<category>/test/<defect_type>/
        self.category_dir = os.path.join(self.root_path, self.category)
        if not os.path.exists(self.category_dir):
            # Sometimes there is an extra nested folder inside the zip
            self.category_dir = os.path.join(self.root_path, "mvtec_anomaly_detection", self.category)
            if not os.path.exists(self.category_dir):
                # Another potential nested structure
                self.category_dir = os.path.join(self.root_path, os.listdir(self.root_path)[0], self.category)
                
        self.split_dir = os.path.join(self.category_dir, self.split)
        
        self.image_paths = []
        self.mask_paths = []
        self.labels = [] # 0 for normal, 1 for anomalous
        
        self._load_dataset()
        
        # Define default transformations if none are provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        if mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.mask_transform = mask_transform

    def _load_dataset(self):
        # Walk split folder structure
        defect_types = sorted(os.listdir(self.split_dir))
        
        for defect in defect_types:
            defect_dir = os.path.join(self.split_dir, defect)
            if not os.path.isdir(defect_dir):
                continue
                
            img_names = sorted(os.listdir(defect_dir))
            for img_name in img_names:
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(defect_dir, img_name)
                self.image_paths.append(img_path)
                
                # Train only contains 'good' normal images
                if self.split == "train":
                    self.labels.append(0)
                    self.mask_paths.append(None)
                else:
                    # Test split
                    if defect == "good":
                        self.labels.append(0)
                        self.mask_paths.append(None)
                    else:
                        self.labels.append(1)
                        # Ground truth mask naming convention: ground_truth/<defect_type>/<xxx_mask>.png
                        img_name_no_ext = os.path.splitext(img_name)[0]
                        mask_path = os.path.join(
                            self.category_dir, 
                            "ground_truth", 
                            defect, 
                            f"{img_name_no_ext}_mask.png"
                        )
                        # Check fallback names if mask is not found
                        if not os.path.exists(mask_path):
                            mask_path = os.path.join(
                                self.category_dir, 
                                "ground_truth", 
                                defect, 
                                f"{img_name_no_ext}.png"
                            )
                        self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        
        # Load ground truth mask
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self.mask_transform(mask)
            # Threshold to binary 0/1
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            # Normal image has all-zero mask
            mask_tensor = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32)
            
        return image_tensor, mask_tensor, label, img_path

def verify_dataset_structure(root_path):
    """
    Spot-checks the downloaded dataset to verify categories, counts, and mask alignments.
    """
    print(f"Checking dataset structure under: {root_path}")
    subdirs = os.listdir(root_path)
    
    # Handle potentially nested folders from unzipping
    if len(subdirs) == 1 and subdirs[0] not in MVTEC_CATEGORIES and os.path.isdir(os.path.join(root_path, subdirs[0])):
        root_path = os.path.join(root_path, subdirs[0])
        subdirs = os.listdir(root_path)
        
    categories = [d for d in subdirs if os.path.isdir(os.path.join(root_path, d))]
    print(f"Found categories: {categories}")
    print(f"Total category count: {len(categories)} (expected: 15)")
    
    # Spot-check first category for mask alignment
    if len(categories) > 0:
        spot_cat = categories[0]
        test_dir = os.path.join(root_path, spot_cat, "test")
        gt_dir = os.path.join(root_path, spot_cat, "ground_truth")
        
        print(f"Spot-checking category '{spot_cat}':")
        if os.path.exists(test_dir):
            defects = [d for d in os.listdir(test_dir) if d != "good"]
            print(f"  Defect types: {defects}")
            if len(defects) > 0 and os.path.exists(gt_dir):
                spot_defect = defects[0]
                img_files = os.listdir(os.path.join(test_dir, spot_defect))
                if len(img_files) > 0:
                    spot_img = img_files[0]
                    img_name_no_ext = os.path.splitext(spot_img)[0]
                    mask_path = os.path.join(gt_dir, spot_defect, f"{img_name_no_ext}_mask.png")
                    print(f"  Verifying test image: {os.path.join(test_dir, spot_defect, spot_img)}")
                    print(f"  Expected mask file: {mask_path}")
                    if os.path.exists(mask_path):
                        print("  [SUCCESS] Ground truth mask exists and is correctly aligned!")
                    else:
                        print("  [WARNING] Ground truth mask file not found in expected path.")
    return root_path
