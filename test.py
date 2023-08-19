import os
import cv2
import json
from data.AMIGOS import check_dir

downsample = 8
root_path = 'data/face_segments'
dst_path = 'data/face_segments_processed_{}'.format(downsample)
check_dir(dst_path)

c = 1
segment_paths = []
for class_name in os.listdir(root_path):
    if '_face' in class_name:
        class_path = os.path.join(root_path, class_name)
        dst_class_path = os.path.join(dst_path, class_name)
        check_dir(dst_class_path)
        for filename in os.listdir(class_path):
            if '_face' in filename:
                segment_path = os.path.join(class_path, filename)
                dst_segment_path = os.path.join(dst_class_path, filename)
                check_dir(dst_segment_path)
                for segment in os.listdir(segment_path):
                    print('process  ', filename, segment)
                    frames_path = os.path.join(segment_path, segment)
                    dst_frames_path = os.path.join(dst_segment_path, segment)
                    check_dir(dst_frames_path)

                    frames = [f for f in os.listdir(frames_path) if '.jpg' in f]
                    frames.sort()
                    # downsampling
                    frames = [frames[idi] for idi in range(len(frames)) if (idi % downsample) == 0]
                    img = {'segmented_frames': [cv2.cvtColor(cv2.imread(os.path.join(frames_path, f)), cv2.COLOR_BGR2RGB).tolist() for f in frames]}
                    json_filename = segment + '.json'
                    with open(os.path.join(dst_frames_path, json_filename), 'w') as f:
                        json.dump(img, f)
                    c += 1
print(c)