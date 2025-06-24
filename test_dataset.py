import os

train_dir = 'dataset/train'
required_extensions = ['_sat.jpg', '_mask.png', '_mask.mat']

missing_files = []
for file in os.listdir(train_dir):
    if file.endswith('_sat.jpg'):
        base = file[:-8]  # removes '_sat.jpg'
        for ext in required_extensions:
            if not os.path.exists(os.path.join(train_dir, base + ext)):
                missing_files.append(base + ext)

if missing_files:
    print("MISSING FILES:")
    for f in missing_files:
        print(f)
else:
    print("All required files are present!")