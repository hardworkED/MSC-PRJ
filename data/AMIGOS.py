import os
import subprocess
import glob
import shutil
import json
import scipy
import torch
import cv2
from matplotlib import pyplot as plt
from scipy.signal import resample
from facenet_pytorch import MTCNN as FMTCNN


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

# special for some Exp2 videos in [N23, N25, N13, N31, N26, N30] due to H264 format error with ffmpeg
def split_frames(root_path):
    total_frames = len([f for f in os.listdir(root_path) if '.jpg' in f])
    vid_segments = [[i for i in range(1, 501)]] + [[i for i in range(j, j + 500)] for j in range(126, total_frames + 1 - 500, 500)] + [[i for i in range(total_frames - 500 + 1, total_frames + 1)]]
    for i in range(1, len(vid_segments) + 1):
        dst_frames_path = os.path.join(root_path, str(i))
        check_dir(dst_frames_path)
        c = 1
        for frame in sorted(vid_segments[i - 1]):
            shutil.copyfile(os.path.join(root_path, 'image_{:05d}.jpg'.format(frame)), os.path.join(dst_frames_path, 'image_{:05d}.jpg'.format(c)))
            c += 1
    for img in glob.glob(os.path.join(root_path, '*.jpg')):
        os.remove(img)

# convert video to frames with segmentation
def vid_to_frames(root_path, dst_dir):
    check_dir(dst_dir)
    for class_name in os.listdir(root_path):
        class_path = os.path.join(root_path, class_name)
        # removing user_id=[8, 24, 28] due to insufficient arousal/valence data
        if not os.path.isdir(class_path) or 'P08' in class_path or 'P24' in class_path or 'P28' in class_path:
            continue
        
        dst_class_path = os.path.join(dst_dir, class_name)
        check_dir(dst_class_path)
    
        for filename in os.listdir(class_path):
            print('Segmenting: ' + filename)
            # removing [P11_18, P13_58, P39_10] for different number of segments from stated
            if '.mov' not in filename or 'P11_18' in filename or 'P13_58' in filename or 'P39_10' in filename:
                continue
            # special for video_id=20 
            vid_len_process = 19 if '_20_' in filename else 20
    
            vid_path = os.path.join(class_path, filename)
            vid_len = get_vid_len(vid_path)
            if not vid_len:
                continue
            if '_N23_' in class_path or '_N25_' in class_path or '_N13_' in class_path or '_N31_' in class_path or '_N26_' in class_path or '_N30_' in class_path:
                dst_vid_path = os.path.join(dst_class_path, os.path.splitext(filename)[0])
                if not check_dir(dst_vid_path):
                    continue
                subprocess.call('ffmpeg -ss 0 -i {} -vf setpts=PTS-STARTPTS,fps=25 -q:v 5 -f image2 "{}/image_%05d.jpg"'.format(vid_path, dst_vid_path),
                    shell=True)
                split_frames(dst_vid_path)
            else:
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

# downsampling ecg to match video fps rate
def ecg_segmentation(file_path):
    with open(file_path, 'r') as f:
        dt = json.load(f)
    freq = 128
    for user_id in dt.keys():
        for video_id in dt[user_id].keys():
            segments = {'ECG_L': {}, 'ECG_R': {}}
            total = len(dt[user_id][video_id]['ECG_L'])
            pts = [[0, freq * 20]] + [[i, i + freq * 20] for i in range(5 * freq, total - freq * 20, freq * 20)] + [[total - freq * 20, total]]
            c = 1
            for pt in pts:
                segments['ECG_L'][str(c)] = dt[user_id][video_id]['ECG_L'][pt[0]:pt[1]]
                segments['ECG_R'][str(c)] = dt[user_id][video_id]['ECG_R'][pt[0]:pt[1]]
                c += 1
            dt[user_id][video_id]['ECG_L'] = segments['ECG_L']
            dt[user_id][video_id]['ECG_R'] = segments['ECG_R']
    save_file_path = os.path.splitext(file_path)[0]
    with open('{}_{}.json'.format(save_file_path, 'segmented'), 'w') as f:
        json.dump(dt, f)

# not recommended
# downsampling ecg to match video fps rate
def ecg_downsampling(file_path, new_freq=25):
    with open(file_path, 'r') as f:
        dt = json.load(f)
    original_freq = 128
    factor = original_freq / new_freq
    for user_id in dt.keys():
        for video_id in dt[user_id].keys():
            dt[user_id][video_id]['ECG_L'] = resample(dt[user_id][video_id]['ECG_L'], int(len(dt[user_id][video_id]['ECG_L']) // factor)).tolist()
            dt[user_id][video_id]['ECG_R'] = resample(dt[user_id][video_id]['ECG_R'], int(len(dt[user_id][video_id]['ECG_R']) // factor)).tolist()
    save_file_path = os.path.splitext(file_path)[0]
    with open('{}_{}.json'.format(save_file_path, new_freq), 'w') as f:
        json.dump(dt, f)

# crop face from frames and resize to 224 x 224
class FastMTCNN(object):
    def __init__(self, stride, resize=1, *args, **kwargs):
        self.stride = stride
        self.resize = resize
        self.mtcnn = FMTCNN(*args, **kwargs)
        
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
            torch.cuda.empty_cache()

def count_frames(root_path):
    dt = {}
    for class_name in os.listdir(root_path):
        class_path = os.path.join(root_path, class_name)
        dt[class_name] = {}
        for filename in os.listdir(class_path):
            frames_path = os.path.join(class_path, filename)
            dt[class_name][filename] = {}
            for segment_frames in os.listdir(frames_path):
                segment_frames_path = os.path.join(frames_path, segment_frames)
                dt[class_name][filename][segment_frames] = len(os.listdir(segment_frames_path))
    return dt

# to exclude video where faces not detected in 10% of frames
def ignore_mov(vids_dir, face_dir):
    ret = []
    vids = count_frames(vids_dir)
    face = count_frames(face_dir)
    for p in sorted(face.keys()):
        for q in sorted(face[p].keys()):
            k = [[face[p][q][r], vids[p][q][r]] for r in sorted(face[p][q].keys())]
            k = list(zip(*k))
            face_sum = sum(k[0])
            vids_sum = sum(k[1])
            if face_sum < (0.9 * vids_sum):
                ret.append(q)
    return ret

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
                                
