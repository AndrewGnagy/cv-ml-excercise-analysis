import tensorflow as tf
import cv2
import numpy as np
import posenet
from pose import Pose
import pickle

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
                #new_data_coords = np.asarray(new_data_coords).reshape(17,2)
                coord_array.append(new_data_coords)
                horizontal_stack = np.hstack((img, input_black_image))
                cv2.imshow("Side-by-side", horizontal_stack)
                cv2.waitKey(1)
            else:
                break
        cap.release()
        
        cv2.destroyAllWindows()
        # print(len(coord_array))
        # print(i)
        # print(video)
        return coord_array

def play_video_for_show(video):
    pose = Pose()
    
    with tf.compat.v1.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(101, sess)
        
        cap = cv2.VideoCapture(video)
        if cap.isOpened() is False:
            print("error in opening video")
        while cap.isOpened():
            ret, img = cap.read()
            if ret:
                img = cv2.resize(img,(372,495))            
                position_data,input_black_image = pose.getpoints_vis(img,sess,model_cfg,model_outputs)[0:34]
                horizontal_stack = np.hstack((img, input_black_image))
                cv2.imshow("Side-by-side", horizontal_stack)
                cv2.waitKey(1)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

def format_and_process(video, out):
    pickle.dump({'train': [], 'labels': [], 'max_len': 0}, open(out, 'wb'))
    data = pickle.load(open(out, 'rb'))
    coord_array = process_video(video)
    coord_array = np.asarray(coord_array).flatten().astype("float32")
    data['train'].append(coord_array)
    data['labels'].append(1)
    data['max_len'] = max(len(coord_array), data['max_len'])
    pickle.dump(data, open(out, 'wb'))

#process_video(args["video"], args["out"])
# max_len = 0
# for x in range(1, 6):
#     coord_array = process_video("squats_good_form_" + str(x) + ".mp4")
#     coord_array = np.asarray(coord_array).flatten().astype("float32")
#     data['train'].append(coord_array)
#     data['labels'].append(1)
#     max_len = max(max_len, len(coord_array))
#     coord_array = process_video("squats_bad_form_" + str(x) + ".mp4")
#     coord_array = np.asarray(coord_array).flatten().astype("float32")
#     data['train'].append(coord_array)
#     data['labels'].append(0)
#     max_len = max(max_len, len(coord_array))
# data['max_len'] = max_len
# print(data)
# pickle.dump(data, open("datafile-jjacks-flat.data", 'wb'))

#format_and_process('jjacks_good_form.mp4', 'jjacks-flat.data')
#play_video_for_show('videos/squats_good_form_1.mp4')