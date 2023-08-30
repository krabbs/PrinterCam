# PrinterCam
A python based software for 3D-Printer observation with multiple cameras

GENERAL INFORMATION:
This project is still in a very early phase. The current release is for better collaboration. Current source code is still poor quality, but functional but good. 

Current features:

Frontend:

-Two video streams in LowQuality with focus on high FPS and low latency <br />
-Display of HQ photos OnDemand <br />
-FocusView (switching between FPS modes of both streams) <br />
-Start / stop PrusaConnect CameraShare <br />
-Start / stop Timelapse <br />
-Visualization of Service Status <br />
-Automatic Day / Night colors according clients “DarkMode” setting. <br />

Backend:

-Automatic initialisation of cameras by name (UDEV alternative) <br />
-asynchronous client management (allows streaming to multiple clients with Ptython Flask) <br />
-Multi threading (one thread per camera, one thread per client. Camera threads send events to clients as soon as a new frame is available, results in low CPU load) <br />
-Intelligent timelapse (series of images in variable time intervals, but a new image for Timeseries video is stored that most closely resembles the old one. This is to prevent fidgeting on the video. I use SSIM or MSE to compare images and decide which one is the best) <br />
-Send images to PrusaConnect <br />
-Provide a merged image from both cameras (merged vertically) <br />
-Provide individual images in full resolution <br />
-Change camera resolution <br />
-Provide JSON API (output status of services, camera, PrusaConnect, timelapse). <br />
-Auto turn on as soon as somebody connects <br />
-Auto turn off camera if nobody is watching <br />
-(Backend based on FLASK and OPENCV) <br />

Hardware:

-Raspberry Pi zero 2 w <br />
-Two cameras (CSI camera, USB camera)

SETUP:

You need your TOKENS for the PRUSA-CAM from PrusaConnect. <br />
In addition the source for opencv has to be defined. The source path is searched based on the usb camera name. this is necessary without udev because the source path can change when booting the hw. For a CSI camera the name is usually "camera0".
They have to be placed in an .env file: <br />
FINGERPRINT=XXXXXXXXXXXXXXXXXXXXXXXXXXX <br />
TOKEN=XXXXXXXXXXXXXXXX <br />
camera1_name=WEBCAM NAME 1 <br />
camera2_name=camera0 <br />
