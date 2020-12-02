from model_evaluate import give_prediction
from keypoints_from_video_flat import format_and_process, play_video_for_show

format_and_process("videos/squats_good_form_1.mp4", "data/squats-good-data1.data")
print("Results: " + give_prediction("data/squats-good-data1.data"))

format_and_process("videos/squats_bad_form_1.mp4", "data/squats-bad-data1.data")
print("Results: " + give_prediction("data/squats-bad-data1.data"))