#Project P.E.T.E

Basically runs a version of yolov7 on two cameras that shows the camera input + yolov7 bounding boxes on a GUI checklist thing

Its also runs on a Jetson Orin Nano. 

Note: a lot of stuff is hardcoded due to basically trial and error to get this stubborn ass board to run this thing

Instructions:
- Launch raspi's gstreamer pipeline (1296x972 for max viewing angle)
raspivid -t 0 -h 972 -w 1296 -fps 25 -hf -b 2000000 -o - | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=1 pt=96 ! gdppay ! tcpserversink sync=0 host=192.168.0.248 port=5000

- Launch this script:
python GUI_detect.py --weights=yolov7-tiny.pt --source=sources.txt --nosave --view-img --no-trace

Quirks with this setup:
1. Raspi will freeze and crash for no reason sometimes (often happens after this program terminates)
HACK: Stop gstreamer pipeline before script terminates
SHTF: REISUB board thru micro USB + keyboard

2. On-board camera seems to only run properly with V4L2 instead of gstreamer with OpenCV
HACK: hardcoded OpenCV flags