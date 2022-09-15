Install packages:
- numpy
- cv2
- pygame
- moviepy

Preparations:
1. Insert your videos in folder 'videos'
2. Insert the video filenames in the csv file 'videoInfo' and save it
3. Empty folder 'frames'

In python environment:
- If camera does not work > change value in 'YET = cv.VideoCapture(1)'
- If frames are presented too short/long > change value of 'image_time = 1'
- Screen dimensions can be changed in 'SCREEN_W = 1000' & 'SCREEN_H = 800'

Data collection:
- Adjust eyetracker
- Run code
- Follow instructions on the screen

Data analysis:
- In the folder of your code (YET\yeta\1\mp4), a csv file is created. In this file you will find all coordinates per video frame.
- In the folder 'frames' the images (.jpg) of these frames are saved.