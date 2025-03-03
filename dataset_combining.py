# import os
# import shutil
# import re

# def organize_brats_dataset(input_dir):
#     # Create the output directory
#     output_dir = os.path.join(os.path.dirname(input_dir), 
#                                f"{os.path.basename(input_dir)}_cleaned")
#     os.makedirs(output_dir, exist_ok=True)

#     # Create subdirectories for modalities
#     modalities = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
#     for modality in modalities:
#         os.makedirs(os.path.join(output_dir, modality), exist_ok=True)

#     # Counters for logging
#     total_files_copied = 0
#     files_skipped = 0

#     # Walk through all subdirectories in the input directory
#     for dataset_folder in os.listdir(input_dir):
#         dataset_path = os.path.join(input_dir, dataset_folder)
        
#         # Skip if not a directory
#         if not os.path.isdir(dataset_path):
#             continue

#         # Walk through patient folders in each dataset
#         for patient_folder in os.listdir(dataset_path):
#             patient_path = os.path.join(dataset_path, patient_folder)
            
#             # Skip if not a directory
#             if not os.path.isdir(patient_path):
#                 continue

#             # Look for files matching BraTS naming pattern
#             patient_files = os.listdir(patient_path)
            
#             for modality in modalities:
#                 # Try to find files for each modality
#                 matching_files = [
#                     f for f in patient_files 
#                     if f.startswith(patient_folder) and 
#                     modality.lower().replace('w', '').replace('ce', '') in f.lower()
#                 ]

#                 for filename in matching_files:
#                     src_path = os.path.join(patient_path, filename)
                    
#                     # Determine the correct modality folder
#                     if modality == 'FLAIR' and 'flair' in filename.lower():
#                         dst_modality = 'FLAIR'
#                     elif modality == 'T1w' and 't1.' in filename.lower():
#                         dst_modality = 'T1w'
#                     elif modality == 'T1wCE' and 't1ce' in filename.lower():
#                         dst_modality = 'T1wCE'
#                     elif modality == 'T2w' and 't2.' in filename.lower():
#                         dst_modality = 'T2w'
#                     else:
#                         continue

#                     # Create destination path
#                     dst_path = os.path.join(output_dir, dst_modality, filename)
                    
#                     # Copy the file
#                     try:
#                         shutil.copy2(src_path, dst_path)
#                         total_files_copied += 1
#                         print(f"Copied: {src_path} -> {dst_path}")
#                     except Exception as e:
#                         files_skipped += 1
#                         print(f"Error copying {src_path}: {e}")

#     # Print summary
#     print("\n--- Copying Complete ---")
#     print(f"Total files copied: {total_files_copied}")
#     print(f"Files skipped due to errors: {files_skipped}")

# def main():
#     # Specify the input directory (BraTS2021_TrainingSet)
#     input_dir = "/Users/Viku/Datasets/Medical/BraTs-2021/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_TrainingSet"
    
#     # Validate input directory
#     if not os.path.isdir(input_dir):
#         print("Invalid directory path. Please provide a valid directory.")
#         return
    
#     # Organize the dataset
#     organize_brats_dataset(input_dir)
#     print("Dataset organization complete!")

# if __name__ == "__main__":
#     main()

import os

def print_folder_structure(root_dir, indent=0):
    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):  # Only consider directories
            print(" " * indent + "|-- " + item)
            print_folder_structure(item_path, indent + 4)


dataset_path = "/Users/Viku/Datasets/Medical/BraTs-2021/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_TrainingSet"  # Change this to your dataset's root directory
print_folder_structure(dataset_path)
