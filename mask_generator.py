import os
from glob import glob
from pathlib import Path
import re

import numpy as np
import cv2
# import openslide
import xml.etree.ElementTree as ET
from skimage import draw
from skimage.segmentation import find_boundaries

try:
    import tifffile
except ImportError:
    tifffile = None

def canonical_image_id(stem: str) -> str:
    # Handles ..._1, ...-1, ..._001
    m = re.match(r"^(.*?)[_-](\d+)$", stem)
    if not m:
        return stem
    prefix, num = m.group(1), int(m.group(2))
    return f"{prefix}_{num:03d}"

# Path where MoNuSAC training data lives
data_path = Path("Training_MoNuSAC_images_and_annotations")

# Where to save generated masks
destination_path = Path("MoNuSAC_outputs")
mask_root = destination_path / "masks"

# create output directories
(mask_root / "ternary").mkdir(parents=True, exist_ok=True)
(mask_root / "semantic_4class").mkdir(parents=True, exist_ok=True)
(mask_root / "instances_all").mkdir(parents=True, exist_ok=True)
(mask_root / "instances" / "epithelial").mkdir(parents=True, exist_ok=True)
(mask_root / "instances" / "lymphocyte").mkdir(parents=True, exist_ok=True)
(mask_root / "instances" / "macrophage").mkdir(parents=True, exist_ok=True)
(mask_root / "instances" / "neutrophil").mkdir(parents=True, exist_ok=True)

# Get patient directories
patients = [
    data_path / d
    for d in os.listdir(data_path)
    if d.startswith("TCGA-") and (data_path / d).is_dir()
]

CLASS_ID = {"Epithelial": 1, "Lymphocyte": 2, "Neutrophil": 3, "Macrophage": 4}

VALID_LABELS = set(CLASS_ID)

def parse_label_from_annotation(ann_el):
    # ImageScope XML often has: Annotation -> Attributes -> Attribute Name = "Epithelial"
    for attr in ann_el.iter("Attribute"):
        name = attr.attrib.get("Name")
        if name in VALID_LABELS:
            return name
    return None

def iter_region_vertices(region_el):
    # Region -> Vertices -> Vertex (X,Y)
    verts_parent = region_el.find(".//Vertices")
    if verts_parent is None:
        return None
    coords = []
    for v in verts_parent.findall(".//Vertex"):
        x = float(v.attrib["X"])
        y = float(v.attrib["Y"])
        coords.append((x, y))
    if len(coords) < 3:
        return None
    return np.asarray(coords, dtype=np.float32) # (N, 2) as (x, y)

for patient_loc in patients:
    # iterate over .tif images
    sub_images = sorted(glob(str(patient_loc / "*.tif")))
    print("TIF count:", len(sub_images), "in", patient_loc.name)

    for sub_image_loc in sub_images:
        sub_image_loc = Path(sub_image_loc)
        image_id = canonical_image_id(sub_image_loc.stem)
        print(patient_loc.name, image_id)

        # read image to get dimensions
        img = cv2.imread(sub_image_loc)
        if img is None:
            print("Could not read image:", sub_image_loc)
            continue
        h, w = img.shape[:2]

        # init masks
        inst = {lbl: np.zeros((h, w), dtype=np.uint16) for lbl in VALID_LABELS}
        next_id = {lbl: 0 for lbl in VALID_LABELS}
        semantic = np.zeros((h, w), dtype=np.uint8)

        # paired xml
        xml_path = sub_image_loc.with_suffix(".xml")
        if not xml_path.exists():
            print("Missing XML:", xml_path)
            continue
        root = ET.parse(str(xml_path)).getroot()

        # ImageScope XML, nuclei of one class grouped under one Annotation element
        # Annotation label applies to all its Regions
        for ann in root.findall(".//Annotation"):
            label = parse_label_from_annotation(ann)
            if label is None:
                continue

            for region in ann.findall(".//Region"):
                coords = iter_region_vertices(region)
                if coords is None:
                    continue
                rr, cc = draw.polygon(coords[:, 1], coords[:, 0], shape=(h, w))
                next_id[label] += 1
                inst[label][rr, cc] = next_id[label]
                semantic[rr, cc] = CLASS_ID[label]

        # compose all_instances for boundary
        all_instances = np.zeros((h, w), dtype=np.uint32)
        offset = 0
        for label in sorted(VALID_LABELS):
            m = inst[label].astype(np.uint32)
            m[m > 0] += offset
            all_instances[m > 0] = m[m > 0]
            offset = int(all_instances.max())

        # save gloabl instance map (unique IDs across all classes)
        max_id = int(all_instances.max())
        if max_id <= 65535:
            all_instances_to_save = all_instances.astype(np.uint16)
        else:
            if tifffile is None:
                raise ValueError(
                    f"all_instances max_id={max_id} exceeds uint16 and tifffile is not installed. "
                    f"Instead via: pip install tifffile"
                )
            tifffile.imwrite(str(mask_root / "instances_all" / f"{image_id}.tif"), all_instances)
        
        cv2.imwrite(str(mask_root / "instances_all" / f"{image_id}.tif"), all_instances_to_save)
        
        inside = (all_instances > 0).astype(np.uint8)
        boundary = find_boundaries(all_instances, mode="inner").astype(np.uint8)

        ternary = np.zeros((h, w), dtype = np.uint8)
        ternary[inside == 1] = 1
        ternary[boundary == 1] = 2

        cv2.imwrite(str(mask_root / "instances" / "epithelial" / f"{image_id}.tif"), inst["Epithelial"])
        cv2.imwrite(str(mask_root / "instances" / "lymphocyte" / f"{image_id}.tif"), inst["Lymphocyte"])
        cv2.imwrite(str(mask_root / "instances" / "macrophage" / f"{image_id}.tif"), inst["Macrophage"])
        cv2.imwrite(str(mask_root / "instances" / "neutrophil" / f"{image_id}.tif"), inst["Neutrophil"])

        cv2.imwrite(str(mask_root / "semantic_4class" / f"{image_id}.tif"), semantic)
        cv2.imwrite(str(mask_root / "ternary" / f"{image_id}.tif"), ternary)