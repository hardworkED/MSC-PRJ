
from AMIGOS import vid_to_frames
import os

root_path = 'data/original_vids'
dst_dir = 'data/vids_segments'

# convert video into frames
vid_to_frames(root_path, dst_dir)