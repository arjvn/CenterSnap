import cv2
import numpy as np
import os
import glob
import pickle
import zstandard as zstd
from CenterSnap.simnet.lib.datapoint import Stereo, Panoptic

class DatasetCreator:
    def __init__(self, left_image_path, right_image_path, depth_image_path, mask_path, scene_name):
        self.left_image_path = left_image_path
        self.right_image_path = right_image_path
        self.depth_image_path = depth_image_path
        self.mask_path = mask_path
        self.scene_name = scene_name
        self.output_path = os.path.join(os.path.dirname(self.left_image_path), "..", f"{scene_name}_dataset")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def create_dataset(self):
        # mask_files = glob.glob(self.mask_path + "*.npy*")
        mask_files = os.listdir(self.mask_path)
        print(f"Found {len(mask_files)} mask files | {self.mask_path}")
        for mask_file in mask_files:
            self.process_mask(mask_file)

    def process_mask(self, mask_file):
        mask_id = self.extract_mask_id(mask_file)
        left_color_image, depth_image, right_color_image, segmentation_mask = self.load_images_and_mask(mask_id)
        panoptic_instance = self.create_panoptic_instance(left_color_image, right_color_image, depth_image, segmentation_mask)
        self.save_instance(panoptic_instance)

    @staticmethod
    def extract_mask_id(mask_file):
        return mask_file.split('binary_overlay_color_image_')[1].split('.png')[0]

    def load_images_and_mask(self, mask_id):
        print(f"Processing mask: {mask_id}")
        print(f"left_image_path: {os.path.join(self.left_image_path, f'color_image_{mask_id}.png')}")
        print(f"depth_image_path: {os.path.join(self.depth_image_path, f'depth_image_{mask_id}.png')}")
        
        left_color_image = cv2.imread(os.path.join(self.left_image_path, f"color_image_{mask_id}.png"))
        depth_image = cv2.imread(os.path.join(self.depth_image_path, f"depth_image_{mask_id}.png"), cv2.IMREAD_UNCHANGED)
        segmentation_mask = np.load(os.path.join(self.mask_path, f"binary_overlay_color_image_{mask_id}.png.npy")).astype(np.float32)
        depth_image = (depth_image / 1000.0).astype(np.float32)
        right_color_image = depth_image  # Or however you wish to handle right image
        segmentation_mask[segmentation_mask == 10] = 1  # Example processing step
        return left_color_image, depth_image, right_color_image, segmentation_mask

    def create_panoptic_instance(self, left_color, right_color, depth, segmentation):
        return Panoptic(
            stereo=Stereo(left_color=left_color, right_color=right_color),
            depth=depth,
            segmentation=segmentation,
            object_poses=[],
            boxes=[],
            detections=[],
            keypoints=[],
            instance_mask=None,
            scene_name=self.scene_name
        )

    def save_instance(self, panoptic_instance):
        serialized_data = pickle.dumps(panoptic_instance)
        compressed_data = zstd.ZstdCompressor().compress(serialized_data)
        output_file_path = os.path.join(self.output_path, f"{panoptic_instance.uid}.pickle.zstd")
        with open(output_file_path, "wb") as output_file:
            output_file.write(compressed_data)
        print(f"Data saved to {output_file_path}")

def run(left_color_image_path, right_color_image_path, depth_image_path, segmentation_mask_path, scene_name):

    dataset_creator = DatasetCreator(
        left_color_image_path,
        right_color_image_path,
        depth_image_path,
        segmentation_mask_path,
        scene_name
    )
    dataset_creator.create_dataset()

if __name__ == "__main__":
    left_color_image_path = '/path/to/left/images/'
    right_color_image_path = '/path/to/right/images/'  # Adjust if you have separate right images
    depth_image_path = '/path/to/depth/images/'
    segmentation_mask_path = '/path/to/mask/images/'
    scene_name = "avocado_scene_test1"
    run(left_color_image_path, right_color_image_path, depth_image_path, segmentation_mask_path, scene_name)