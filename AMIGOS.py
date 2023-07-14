import os
import subprocess
import json
import scipy
from facenet_pytorch import MTCNN
import torch
import cv2
from matplotlib import pyplot as plt

# user_id=[8, 24, 28] removed for missing arousal and valence

# check if folder exist
def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        return True
    return False

# get video length
def get_vid_len(filename: str) -> float:
    try:
        duration = subprocess.check_output('ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{}"'.format(filename),
            shell=True)
        return int(float(duration))
    except Exception as e:
        print(e)
        return

# convert video to frames with segmentation
def vid_to_frames(root_path, dst_dir):
    check_dir(dst_dir)
    for class_name in os.listdir(root_path):
        class_path = os.path.join(root_path, class_name)
        # removing user_id=[8,24,28] due to insufficient arousal/valence data
        if not os.path.isdir(class_path) or 'P08' in class_path or 'P24' in class_path or 'P28' in class_path:
            continue
        
        dst_class_path = os.path.join(dst_dir, class_name)
        check_dir(dst_class_path)
    
        for filename in os.listdir(class_path):
            print('Segmenting: ' + filename)
            if '.mov' not in filename or 'P11_18' in filename or 'P13_58' in filename or 'P39_10' in filename:
                continue
            # special for video_id=20 
            vid_len_process = 19 if '_20_' in filename else 20
    
            vid_path = os.path.join(class_path, filename)
            vid_len = get_vid_len(vid_path)
            if not vid_len:
                continue
            vid_segments_config = ['-ss 0 -t 20'] + ['-ss {} -t 20'.format(i) for i in range(5, vid_len - vid_len_process, 20)] + ['-sseof -20']
            for i in range(len(vid_segments_config)):
                print('Segment: ' + filename + ' ' + str(i))
                dst_frames_path = os.path.join(dst_class_path, os.path.splitext(filename)[0], str(i+1))
                if not check_dir(dst_frames_path):
                    continue
                # video at 25 fps
                subprocess.call('ffmpeg {} -i {} -vf setpts=PTS-STARTPTS,fps=25 -q:v 5 -f image2 "{}/image_%05d.jpg"'.format(vid_segments_config[i], vid_path, dst_frames_path),
                    shell=True)

# preprocessed data to json
# arousal/valence value from external_annotations is mean/2
def data_preprocessed(data_dir, data_preprocessed_path, json_filename):
    dt = {}
    for mat_file in os.listdir(data_preprocessed_path):
        mat_folder = os.path.join(data_preprocessed_path, mat_file)
        mat = scipy.io.loadmat(os.path.join(mat_folder, mat_file))
        dt[mat_file] = {str(mat['VideoIDs'][0][i][0]): {
            'ECG_L': mat['joined_data'][0][i].transpose()[15].tolist(),
            'ECG_R': mat['joined_data'][0][i].transpose()[14].tolist(),
            'AV': {int(mat['labels_ext_annotation'][0][i][j][0]): {
                'valence': mat['labels_ext_annotation'][0][i][j][1],
                'arousal': mat['labels_ext_annotation'][0][i][j][2]
            } for j in range(mat['labels_ext_annotation'][0][i].shape[0])}
        } for i in range(mat['VideoIDs'].shape[1])}
    with open(os.path.join(data_dir, json_filename), 'w') as f:
        json.dump(dt, f)

# crop face from frames and resize to 224 x 224
class FastMTCNN(object):
    def __init__(self, stride, resize=1, *args, **kwargs):
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                    for f in frames
            ]

        # split frames
        frames_split = []
        c = 0
        while c < len(frames):
            if c + 100 <  len(frames):
                frames_split.append(frames[c : c+100])
            else:
                frames_split.append(frames[c : len(frames)])
            c += 100

        faces = []
        for fs in frames_split:
            boxes, probs = self.mtcnn.detect(fs)

            for i, frame in enumerate(fs):
                if boxes[i] is not None and len(boxes[i]) > 0:
                    box = [int(b) for b in boxes[i][0]]
                    faces.append(frame[box[1]:box[3], box[0]:box[2]])
                else:
                    faces.append([])
            
            del boxes, probs
            torch.cuda.empty_cache() 
        return faces

def face_detection_fm(root_path, dst_dir, device, filter=''):
    check_dir(dst_dir)
    fast_mtcnn = FastMTCNN(
        # thresholds=[0.8, 0.9, 0.9],
        selection_method='probability',
        stride=1,
        resize=1,
        margin=0,
        device=device
    )

    class_names = os.listdir(root_path)
    if filter:
        class_names = [class_name for class_name in class_names if filter in class_name]
    for class_name in class_names:
        class_path = os.path.join(root_path, class_name)
        dst_class_path = os.path.join(dst_dir, class_name)
        check_dir(dst_class_path)
        for filename in os.listdir(class_path):
            frames_path = os.path.join(class_path, filename)
            dst_frames_path = os.path.join(dst_class_path, filename)
            check_dir(dst_frames_path)
            for segment_frames in os.listdir(frames_path):
                segment_frames_path = os.path.join(frames_path, segment_frames)
                print('Processing: ', segment_frames_path)
                dst_segment_frames_path = os.path.join(dst_frames_path, segment_frames)
                if not check_dir(dst_segment_frames_path):
                    continue
                frame_paths, dst_frame_paths = list(zip(*[[os.path.join(segment_frames_path, img_name), os.path.join(dst_segment_frames_path, img_name)] for img_name in os.listdir(segment_frames_path) if '.jpg' in img_name]))
                imgs = [cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB) for frame_path in frame_paths]
                faces = fast_mtcnn(imgs)
                for idx in range(len(faces)):
                    if len(faces[idx]) > 0 and 0 not in faces[idx].shape:
                        plt.imsave(dst_frame_paths[idx], cv2.resize(faces[idx], (224, 224), interpolation=cv2.INTER_CUBIC))
                del imgs, faces

# processing too slow
from mtcnn import MTCNN
def face_detection(root_path, dst_dir):
    check_dir(dst_dir)
    mtcnn = MTCNN()
    ds = []
    for class_name in os.listdir(root_path):
        class_path = os.path.join(root_path, class_name)
        dst_class_path = os.path.join(dst_dir, class_name)
        check_dir(dst_class_path)
        
        for filename in os.listdir(class_path):
            frames_path = os.path.join(class_path, filename)
            dst_frames_path = os.path.join(dst_class_path, filename)
            check_dir(dst_frames_path)
            for segment_frames in os.listdir(frames_path):
                segment_frames_path = os.path.join(frames_path, segment_frames)
                dst_segment_frames_path = os.path.join(dst_frames_path, segment_frames)
                if not check_dir(dst_segment_frames_path):
                    continue
                for img_name in os.listdir(segment_frames_path):
                    if '.jpg' in img_name:
                        frame_path = os.path.join(segment_frames_path, img_name)
                        print(frame_path)
                        img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
                        detections = mtcnn.detect_faces(img)
                        if len(detections) > 1:
                            ds.append(frame_path)
                        for detection in detections:
                            if detection['confidence'] > 0.9:
                                plt.imsave(
                                    os.path.join(dst_segment_frames_path, img_name), 
                                    cv2.resize(
                                        img[detection['box'][1]:detection['box'][1] + detection['box'][3], detection['box'][0]:detection['box'][0] + detection['box'][2]],
                                        (224, 224),
                                        interpolation=cv2.INTER_CUBIC))
                                
