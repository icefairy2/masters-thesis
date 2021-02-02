import os
from shutil import copyfile, copy

GESTURE_DATASET_FOLDER = 'gesture_dataset'

# delete all depth videos
for root, dirs, files in os.walk(GESTURE_DATASET_FOLDER):
    for file in files:
        if file.startswith('K'):
            os.remove(os.path.join(root, file))

# get all labels for files
f = open(os.path.join(GESTURE_DATASET_FOLDER, 'train_list.txt'), "r")
lines = f.readlines()
file_to_label = {}
for x in lines:
    file_to_label[x.split(' ')[0]] = int(x.split(' ')[2])
f.close()

# sort files by labels
sorted_by_label = sorted(file_to_label.items(), key=lambda kv: kv[1])

# create new folder structure
new_folder = os.path.join(GESTURE_DATASET_FOLDER, 'organized')
os.makedirs(new_folder, exist_ok=True)

for original_path, label in sorted_by_label:
    filename = original_path.replace('train/', '').replace('/', '_')
    label_folder = os.path.join(new_folder, str(label))
    os.makedirs(label_folder, exist_ok=True)

    copy(os.path.join(GESTURE_DATASET_FOLDER, original_path), label_folder)
