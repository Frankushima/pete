# Project P.E.T.E

Basically runs a version of yolov7 on two cameras that shows the camera input + yolov7 bounding boxes on a GUI checklist thing

Its also runs on a Jetson Orin Nano. 

Note: a lot of stuff is hardcoded due to basically trial and error to get this stubborn ass board to run this thing

## Setup for running with video (Read This):
1. Download testing videos from Slack or use ur own
2. Download model weight files from [here](https://drive.google.com/drive/u/1/folders/1_MV9JUMt3BgdXSHATWRznBClBUjr-OBL)
3. Run a test run with default detect:
```bash
python detect.py --weights=yolov7-tiny.pt --source=test_videos/bottomBracketInstall.MOV --nosave --view-img --no-trace
```
4. Run a command similar to this one (This is the actual command you will use from now on):
```bash
python GUI_detect.py --weights=yolov7-tiny.pt --source=test_videos/bottomBracketInstall.MOV --nosave --view-img --no-trace
```

Arguments:
- weights = which model to use
- source = test video file location
- nosave = don't save results
- view-img = needs to be there for the GUI to work
- no-trace = skip tracing model step to make bootup faster

Quirks with this setup:
1. Video plays at slower speed to match detection rate (HACK: made it skip every other frame)

## TODO: 
- Make video skip frames to simulate real time enviroment (DONE)
- Add loading gif (DONE)
- Add manual revert step (DONE)
- Add substeps with pictures for context (WIP)
- Add sensor stuff

## FOR LIVE ONLY 

Instructions for live object detection with two cameras:
- Launch raspi's gstreamer pipeline (1296x972 for max viewing angle)
```bash
raspivid -t 0 -h 972 -w 1296 -fps 25 -hf -b 2000000 -o - | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=1 pt=96 ! gdppay ! tcpserversink sync=0 host=192.168.0.248 port=5000
```

- Launch this script:
```bash
python GUI_detect.py --weights=Demo_Only_B40.pt --source=test_videos/step7.mov --nosave --view-img --no-trace
```

Quirks with this setup:
1. Raspi will freeze and crash for no reason sometimes (often happens after this program terminates)
HACK: Stop gstreamer pipeline before script terminates
SHTF: REISUB board thru micro USB + keyboard

2. If you switch USB devices while the model is loading, it might freeze the board... edge device things
SHTF: REISUB board thru micro USB + keyboard

3. On-board camera seems to only run properly with V4L2 instead of gstreamer with OpenCV
HACK: hardcoded OpenCV flags


## PROGRESS REPORTS

4/13:
- Fixed procedure list scaling issues
- Fixed focus bug on reverting steps
- Added loading gif
- Fixed exiting window not exiting program bug
- Added pictures to steps (WIP)
- Fixed gitignore to make sure stuff is downloaded correctly


## Thoughts

- Could have a "search and match" sequence for the substeps within a step. 
  - Like a state machine? But isn't this already what it's doing right now?