a_first_step_into_our_smart_city implemented in YoloV3

This is a project done by a team of Students from the Addis Ababa Science and Technology University consisting of 5 team members, under the supervision of Professor Dr. Srinivasan Ramasamy  1st degree connection, Professor at RMK ENGINEERING COLLEGE, CHENNAI. Our team:

1) Eyobed Tilaye Kebede
2) Eyuel Tibebu
3) Ezedin Ali
4) Milkessa Oljira
5) Sifan Dinsa


This project requires GPU support to run efficiently. Although it might run on CPU environment, it is not recommended.
The GPU environment used in this project is NVIDIA's platform. This project is also theoretically transferrable to Google Colab. If you have any problems with your environment, please put in an error so that we might fix it promptly.


For the functioning of the project, pre trained weights on the COCO dataset are used. You can find them here: https://pjreddie.com/media/files/yolov3.weights. Download it, then copy the weights to the 'weights' folder found in your clone of this repository and then run the following command from your command line. Be sure to configure your directory to the cloned repository in your local drive.

python load_weights.py



The algorithm accepts two inputs for video: a video stored in your local drive or a webcam. The default setting is a video input from your local drive. To change this, go to line 39 in main.py and comment it out, and uncomment line 40.



This algorithm exports two types of csv files that will later be used in a  prediction algorithm. One is labled per_5_min_data and the other is daily_data. As the name indicates, the algorithm will export the data it observes every 5 mins to one excel file and a daily data containing the daily average, highs, and lows in another file.

Let us know if you have any questions.

We've also provided Sample Videos in an Ethiopian Concept so as to check the algorithms performance with. You will fine two .mp4 files in the current working directory. You can use these or any other video file in any other directory.

We are currently working on an implementation of this concept in Yolov4 and possibly Yolov5. So stay tuned!

 
