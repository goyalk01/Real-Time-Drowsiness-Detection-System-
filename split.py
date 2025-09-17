import os
from sklearn.model_selection import train_test_split

def load_image_paths_and_labels(data_dir):
    """
    Loads image file paths and corresponding labels (0: notdrowsy, 1: drowsy)
    assuming data_dir has two subfolders: 'notdrowsy' and 'drowsy'.
    """
    classes = {'notdrowsy': 0, 'drowsy': 1}
    image_paths = []
    labels = []
    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, filename))
                labels.append(classes[cls])
    return image_paths, labels

# Use portable relative data path
data_directory = "train_data"

image_paths, labels = load_image_paths_and_labels(data_directory)

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels)

print(f'Training samples: {len(train_paths)}')
print(f'Testing samples: {len(test_paths)}')

with open('train.txt', 'w') as f:
    for path, label in zip(train_paths, train_labels):
        f.write(f"{path} {label}\n")

with open('test.txt', 'w') as f:
    for path, label in zip(test_paths, test_labels):
        f.write(f"{path} {label}\n")
