
# MOT_test_Khalilullah
## Data Annotation
I have annotated one video (Test_Lacrosse_1080.mp4) using Darklabel annotation software [darklabel](https://github.com/darkpgmr/DarkLabel). I shall use this video for training after completed a model preparation for tracking.

# Training
### data preparation from annotation data
Use data_prepa_labels_file.py and data_prepa_generate_path_file.py to create label files and path file from the annotated data.
Download pretrained model, configuration file, and associated folders from this link.[models](https://drive.google.com/drive/folders/1myYZUre4hXPoyzKrmbVSAzWwPp0y7lif?usp=sharing)

Then replace the folders in the MOT_test_Khalilullah folder.
run the train.py file: python train.py

# Detection and Tracking
run track.py file: python track.py

# Qualitative results
[results](https://drive.google.com/drive/folders/1PmNKGFo-UP7ZZ5SrTcYdr8VUv9TvGJkB?usp=sharing)
=======
# Multi_object_tracking_vision
Datalink and References    
    Multi-Object Tracking Wiki: https://en.wikipedia.org/wiki/Multiple_object_tracking
    Multi-Object Tracking papers and Codes: https://paperswithcode.com/task/multi-object-tracking
    Learn OpenCV: https://learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/
    MOT Challenge: https://motchallenge.net/

