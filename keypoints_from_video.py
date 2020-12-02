import tensorflow as tf
import cv2
import numpy as np
import posenet
from pose import Pose
from score import Score
import pickle
import argparse

#USAGE : python3 keypoints_from_video.py --video "test.mp4" 
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="video file from which keypoints are to be extracted")
ap.add_argument("-o", "--out", default="data_new.pickle",
    help="The pickle file to dump the coordinate data to")
args = vars(ap.parse_args())


def process_video(video):
    pose = Pose()
    coord_array = []
    
    with tf.compat.v1.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(101, sess)
        
        cap = cv2.VideoCapture(video)
        i = 0
        if cap.isOpened() is False:
            print("error in opening video")
        while cap.isOpened():
            ret, img = cap.read()
            if ret:
                i += 1
                img = cv2.resize(img,(372,495))            
                position_data,input_black_image = pose.getpoints_vis(img,sess,model_cfg,model_outputs)[0:34]
                if not position_data:
                    continue
                new_data_coords = pose.roi(position_data)[0:34]
                #Reshape coordinates as 17x2 matrix
                new_data_coords = np.asarray(new_data_coords).reshape(17,2).astype("float32")
                coord_array.append(new_data_coords)
                cv2.imshow("black", input_black_image)
                cv2.waitKey(1)
            else:
                break
        cap.release()
        
        cv2.destroyAllWindows()
        print(len(coord_array))
        print(i)
        print(video)
        return coord_array

#process_video(args["video"], args["out"])
pickle.dump({'train': [], 'labels': [], 'max_len': 0}, open("datafile.data", 'wb'))
data = pickle.load(open("datafile.data", 'rb'))
max_len = 0
for x in range(1, 6):
	coord_array = process_video("squats_good_form_" + str(x) + ".mp4")
	data['train'].append(np.asarray(coord_array))
	data['labels'].append(1)
	max_len = max(max_len, len(coord_array))
	coord_array = process_video("squats_bad_form_" + str(x) + ".mp4")
	data['train'].append(np.asarray(coord_array))
	data['labels'].append(0)
	max_len = max(max_len, len(coord_array))
data['max_len'] = max_len
print(data)
pickle.dump(data, open("datafile.data", 'wb'))