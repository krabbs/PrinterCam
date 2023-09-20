import cv2
import time
from datetime import datetime
import imutils
import threading
import sys
import os
import re
import numpy as np
import requests
import ffmpeg
import argparse
from skimage.metrics import structural_similarity as ssim
import urllib3
import json

urllib3.disable_warnings()

try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident

class VideoCamera(object):
    max_cams = 3
    timelapse = None
    timelapse_flag = False
    timelapse_folder = None
    prusa = None
    event = [None] * max_cams
    req = [threading.Event()] * max_cams
    unlock = threading.Event()
    thread = [None] * max_cams
    thread_runflag = [False] * max_cams
    DAEMONthread = [None] * max_cams
    DAEMONthread_runflag = [False] * max_cams
    video = [None] * max_cams
    camera_source = [None] * max_cams
    errorCounter = [0] * max_cams
    tStart = [0] * max_cams
    description = ["std"] * max_cams
    rotate = [0] * max_cams
    fps = [10] * max_cams
    targetfps = [10] * max_cams
    lastImage = [b''] * max_cams
    lasttime = [0] * max_cams
    cameraState = [False] * max_cams
    streamCount = [0] * max_cams
    pxl_H = [480] * max_cams
    pxl_V = [640] * max_cams
    pxl_H_hres = [1080] * max_cams
    pxl_V_hres = [1920] * max_cams
    last_access = [0] * max_cams
    errorFrames = [0] * max_cams
    thread_events = True
    read_buffer = True

    def __init__(self, cam, description, rotate=0, warmup=False):
        VideoCamera.unlock.clear()
        self.camera_obj_source = cam
        VideoCamera.camera_source[cam] = cam
        VideoCamera.description[cam] = description
        VideoCamera.rotate[cam] = rotate
        if (VideoCamera.createCam(cam, warmup)):
            VideoCamera.cameraState[cam] = True
            VideoCamera.streamCount[cam] += 1
        else:
            if VideoCamera.lastImage[cam] != b'':
                # vermutlich schon erstellt
                VideoCamera.cameraState[cam] = True
                VideoCamera.streamCount[cam] += 1
                print("Stick to stream")
            else:
                VideoCamera.cameraState[cam] = False
                print("error Vid: " + str(cam))
        VideoCamera.unlock.set()
        #print(VideoCamera.streamCount)

    def __del__(self):
        cam = self.camera_obj_source
        VideoCamera.streamCount[cam] -= 1
        print(VideoCamera.streamCount)
        
    def resetcam(cam1, cam2=None):
      VideoCamera.unlock.clear()
      print("!!!!!!!!!!!resetCams!!!!!!!!!!")
      VideoCamera.stop(cam1, True)
      if cam2 is not None: VideoCamera.stop(cam2, True)
      time.sleep(0.03)
      print("!!!!!!!!!!!restartCams!!!!!!!!!!")
      if (VideoCamera.createCam(cam1, False)):
            VideoCamera.cameraState[cam1] = True
            print("CAM: " + str(cam1) + " restartet")
            VideoCamera.streamCount[cam1] += 1
      if (VideoCamera.createCam(cam2, False)) and (cam2 is not None):
            VideoCamera.cameraState[cam2] = True
            print("CAM: " + str(cam2) + " restartet")
            VideoCamera.streamCount[cam2] += 1
      VideoCamera.unlock.set()
      
    def stop(cam, force=False):
        # cam = self.camera_obj_source
        print("Camera " + str(cam) + " stopping:")
        print(VideoCamera.streamCount)
        if (VideoCamera.streamCount[cam] > 0 and VideoCamera.video[cam] != None) or force:
            VideoCamera.streamCount[cam] = 0
            print("release: " + str(cam))
            VideoCamera.video[cam].release()
            VideoCamera.event[cam].clear()
            VideoCamera.thread_runflag[cam] = False
            VideoCamera.DAEMONthread_runflag[cam] = False
            if VideoCamera.thread[cam] is not None: 
              print("try to gracefuly stop thread")
              VideoCamera.thread[cam].join(timeout=5)
              if VideoCamera.thread[cam].is_alive(): print("warn: thread could not be stopped")
            else:
              print(" thread not active")
            if VideoCamera.DAEMONthread[cam] is not None: 
              print("try to gracefuly stop daemon")
              VideoCamera.DAEMONthread[cam].join(timeout=5)
              if VideoCamera.DAEMONthread[cam].is_alive(): print("warn: daemon could not be stopped")
            else:
              print("daemon not active")
            VideoCamera.thread[cam] = None
            VideoCamera.DAEMONthread[cam] = None
            VideoCamera.video[cam] = None
            VideoCamera.cameraState[cam] = False
            time.sleep(0.03)
            print(VideoCamera.streamCount)
            print("Camera " + str(cam) + " STOPPED:")

    def get_camera(camera_name):
        cam_num = None
        for file in os.listdir("/sys/class/video4linux"):
            real_file = os.path.realpath(
                "/sys/class/video4linux/" + file + "/name")
            with open(real_file, "rt") as name_file:
                name = name_file.read().rstrip()
            if camera_name in name:
                new_cam_num = int(re.search("\d+$", file).group(0))
                found = "FOUND!"
                if cam_num == None:
                    cam_num = new_cam_num
                elif new_cam_num < cam_num:
                    cam_num = new_cam_num
            else:
                found = "      "
            print("{} {} -> {}".format(found, file, name))
        return cam_num

    @classmethod
    def createCam(cls, cam, warmup=False):
        if VideoCamera.video[cam] != None:
            return False
        else:
            VideoCamera.video[cam] = cv2.VideoCapture(cam, cv2.CAP_V4L2)
            print("create Vid: " + str(cam) + " Name:" +
                  VideoCamera.video[cam].getBackendName())
            VideoCamera.tStart[cam] = time.time()
            #VideoCamera.video[cam].set(cv2.CAP_PROP_BUFFERSIZE, 2)
            if VideoCamera.description[cam] == "printer":
                VideoCamera.targetfps[cam] = 15  # 30#24#20#15#10#7#5
                VideoCamera.pxl_V[cam] = 800  # 1920 #1280 #800
                VideoCamera.pxl_H[cam] = 600
            # VideoCamera.video[cam].set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # auto mode
            # VideoCamera.video[cam].set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual mode
            # VideoCamera.video[cam].set(cv2.CAP_PROP_EXPOSURE, 40)
            VideoCamera.video[cam].set(cv2.CAP_PROP_FPS, VideoCamera.targetfps[cam])
            VideoCamera.video[cam].set(
                cv2.CAP_PROP_FRAME_WIDTH, VideoCamera.pxl_V[cam])
            VideoCamera.video[cam].set(
                cv2.CAP_PROP_FRAME_HEIGHT, VideoCamera.pxl_H[cam])
            if warmup:
                print("warm up: " + str(cam))
                for i in range(4):
                    print(".", sep=' ', end='', flush=True)
                    success, image = VideoCamera.video[cam].read()
                    time.sleep(0)
                print("done")
            (grabbed, frame) = VideoCamera.video[cam].read()
            image = './cam' + str(cam) + '.jpg'
            cv2.imwrite(image, frame)
            VideoCamera.cameraState[cam] = True
            # start background frame thread
            if VideoCamera.read_buffer:
                VideoCamera.thread[cam] = threading.Thread(target=VideoCamera._thread_buffer, daemon=False, args=(cam,))
            else:
                VideoCamera.thread[cam] = threading.Thread(target=VideoCamera._thread, daemon=False, args=(cam,))
            VideoCamera.DAEMONthread[cam] = threading.Thread(target=VideoCamera._DAEMONthread, args=(cam,))
            if VideoCamera.thread_events: VideoCamera.event[cam] = threading.Event()
            VideoCamera.last_access[cam] = time.time()
            VideoCamera.thread_runflag[cam] = True
            if VideoCamera.thread[cam].is_alive():
              print("warn: thread not created. already alive")
            else:
              VideoCamera.thread[cam].start()
            VideoCamera.DAEMONthread_runflag[cam] = True
            if VideoCamera.DAEMONthread[cam].is_alive():
              print("warn: daemon not created. already alive")
            else:
              VideoCamera.DAEMONthread[cam].start()
            return True

    def setFPS(cam, fps):
        VideoCamera.targetfps[cam] = fps
        VideoCamera.setProb(cam, cv2.CAP_PROP_FPS, fps)

    def setProb(cam, arg1, arg2):
        if VideoCamera.video[cam] is not None:    
          VideoCamera.unlock.clear()
          if VideoCamera.video[cam].isOpened():
              VideoCamera.video[cam].release()
              VideoCamera.video[cam] = cv2.VideoCapture(cam, cv2.CAP_V4L2)
              time.sleep(1)
              VideoCamera.video[cam].set(arg1, arg2)
              time.sleep(1)
          else:
              VideoCamera.video[cam].set(arg1, arg2)
          VideoCamera.unlock.set()

    # @classmethod
    def setFPStime(cam, time):
        fps_t = time - VideoCamera.lasttime[cam]
        VideoCamera.lasttime[cam] = time
        VideoCamera.fps[cam] = 0.95*VideoCamera.fps[cam] + 0.05*(1/fps_t)
        # VideoCamera.fps[cam]=(1/fps_t)
        # print("Video" + str(self.cam) + ": " + str('{:.0f}'.format(round(self.fps,0)))+"/"+str(self.targetfps))
        return VideoCamera.fps[cam]

    @staticmethod
    def sum_frame(pic1, pic2):
        if pic1 == None or pic2 == None:
            return b''
        image_array1 = np.frombuffer(pic1, np.uint8)
        image_array2 = np.frombuffer(pic2, np.uint8)
        # Bild mit OpenCV einlesen
        image1 = cv2.imdecode(image_array1, cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(image_array2, cv2.IMREAD_COLOR)
        # Bilder auf gleiche Breite skalieren
        min_width = min(image1.shape[1], image2.shape[1])
        image1 = VideoCamera.image_resize(image1, width=min_width)
        image2 = VideoCamera.image_resize(image2, width=min_width)
        c_image = cv2.vconcat([image1, image2])
        # resized = cv2.resize(c_image, (480, 320), interpolation = cv2.INTER_AREA)
        # cv2.imwrite('./cam_image1.jpg', image1)
        # cv2.imwrite('./cam_image2.jpg', image2)
        cv2.imwrite('./cam_imagem.jpg', c_image)
        ret, jpeg = cv2.imencode('.jpg', c_image)
        return jpeg.tobytes()

    @staticmethod
    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def get_frame(self, ss=False, hres=False):
        cam = self.camera_obj_source
        ident = get_ident()
        VideoCamera.last_access[cam] = time.time()
        if ss:
            print("Single Shot")
            if hres:
                VideoCamera.req[cam].set()
                ss = VideoCamera.frame(cam, TIMEdata=True, FPSdata=False, hres=True)
            else:
                ss = VideoCamera.frame(cam, TIMEdata=True, FPSdata=False)
            return ss
        if VideoCamera.thread_events:
            #print(str(cam)+" Thread request: " + str(ident))
            VideoCamera.req[cam].set()
            time.sleep(0.03)
            #print(str(cam)+" Thread wait: " + str(ident))
            if not VideoCamera.event[cam].wait(1.2): print(str(cam)+" Warn: Thread event fail: " + str(ident))
            #print(str(cam)+" Thread event: " + str(ident))
            VideoCamera.event[cam].clear()
        if VideoCamera.lastImage[cam] == b'':
            time.sleep(0)
            print(str(cam)+" Warn: empty frame. Browser may crash")
            return VideoCamera.lastImage[cam]
        else:
            return VideoCamera.lastImage[cam]

    def frame(cam, TIMEdata=True, FPSdata=True, hres=False):
        # read current frame
        if hres:
            VideoCamera.video[cam].set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            VideoCamera.video[cam].set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            success, img = VideoCamera.video[cam].read()
            VideoCamera.video[cam].set(
                cv2.CAP_PROP_FRAME_WIDTH, VideoCamera.pxl_V[cam])
            VideoCamera.video[cam].set(
                cv2.CAP_PROP_FRAME_HEIGHT, VideoCamera.pxl_H[cam])
        else:
            success, img = VideoCamera.video[cam].read()
        if success:
            VideoCamera.errorFrames[cam] = 0
            if VideoCamera.rotate[cam] == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif VideoCamera.rotate[cam] == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif VideoCamera.rotate[cam] == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            VideoCamera.addData(img, cam, TIMEdata, FPSdata)
            # encode as a jpeg image and return it
            return cv2.imencode('.jpg', img)[1].tobytes()
        else:
            VideoCamera.errorFrames[cam] +=1
            return b''

    def frames(cam, TIMEdata=True, FPSdata=True):
        # print(TIMEdata)
        # print(FPSdata)
        # cam = self.camera_obj_source
        while True:
            # read current frame
            if VideoCamera.video[cam] is not None:
              success, img = VideoCamera.video[cam].read()
              if success:
                  VideoCamera.errorFrames[cam] = 0
                  if VideoCamera.rotate[cam] == 180:
                      img = cv2.rotate(img, cv2.ROTATE_180)
                  elif VideoCamera.rotate[cam] == 90:
                      img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                  elif VideoCamera.rotate[cam] == 270:
                      img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                  VideoCamera.addData(img, cam, TIMEdata, FPSdata)
                  # encode as a jpeg image and return it
                  yield cv2.imencode('.jpg', img)[1].tobytes()
              else:
                  VideoCamera.errorFrames[cam] +=1
                  yield b''
            else:
              yield b''

    @staticmethod
    def addData(image, cam, TIMEdata=True, FPSdata=True):
        # Write some Text
        time_now = datetime.now()
        current_time = time_now.strftime("%H:%M:%S")
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 10)
        fontScale = 0.6
        fontColor = (255, 255, 255)
        thickness = 2
        lineType = 2
        # print(TIMEdata)
        # print(FPSdata)
        if TIMEdata and FPSdata:
            image = cv2.putText(image, current_time + ' FPS: ' + str('{:.0f}'.format(round(VideoCamera.fps[cam], 0)))+"/"+str(VideoCamera.targetfps[cam]),
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)
        elif TIMEdata:
            #print("AddTime")
            cv2.putText(image, 'Time: ' + current_time,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
        return image

    @classmethod
    def _thread(cls, cam):
        """Camera background thread."""
        print('Starting camera thread. For cam: ' +  str(cam))
        skip = 0
        while VideoCamera.thread_runflag[cam]:
              ident = get_ident()
               # print(str(cam)+" Backgroundthread start: " + str(ident))
               # and (VideoCamera.cameraState[cam]):
              if (0.9/(VideoCamera.targetfps[cam]) + VideoCamera.lasttime[cam]) < time.time() or skip > 5:
                    skip = 0
                    # print(str(cam)+" Backgroundthread lock: " + str(ident))
                    if VideoCamera.thread_events:
                        if not VideoCamera.unlock.wait(5):
                          #print(str(cam)+"Warn: Backgroundthread locked. timout for cam: " + str(ident))
                          time.sleep(0)
                        #print(str(cam)+" Backgroundthread unlocked and wait for request: " + str(ident))
                        if not VideoCamera.req[cam].wait(1): 
                          #print(str(cam)+"Warn: Backgroundthread request. timout for cam: " + str(ident))
                          time.sleep(0)
                        VideoCamera.req[cam].clear()
                    frame = cls.frame(cam)
                    if frame != b'':
                        VideoCamera.lastImage[cam] = frame
                        VideoCamera.setFPStime(cam, time.time())
                        #print("Frame of Camera: " +  str(cam) + " FPS:" + str('{:.0f}'.format(round(VideoCamera.fps[cam],0)))+"/"+str(VideoCamera.targetfps[cam]))   
                        if VideoCamera.thread_events: VideoCamera.event[cam].set()
                    else:
                        print(str(cam)+" Backgroundthread SHOT EMPTY " + str(ident))                  
              else:
                    #print("Call to fast for cam " + str(cam))
                    skip += 1
                    left = ( 0.5/VideoCamera.targetfps[cam]+VideoCamera.lasttime[cam])
                    right = time.time()
                    #print(str(left) + "<"+str(right)+"= " + str(right-left) + "or FPS:" + str(1/(right-left)))
                    if VideoCamera.thread_events: VideoCamera.req[cam].set()
              time.sleep(0.03)
        print('Warn: Stopping camera thread. For cam: ' +  str(cam))             
    
    @staticmethod          
    def prusa_send(image, HTTP_URL, FINGERPRINT, TOKEN):
        response = VideoCamera.upload_image(HTTP_URL, FINGERPRINT, TOKEN, image)
        print("prusa upload"+ str(response))


    def upload_image(http_url, fingerprint, token, image):
        response = requests.put(
            http_url,
            headers={
                "accept": "*/*",
                "content-type": "image/jpg",
                "fingerprint": fingerprint,
                "token": token,
            },
            data=image,
            stream=True,
            verify=False,
        )
        return response
        
    @classmethod
    def _thread_buffer(cls, cam):
        """Camera background thread."""
        print('Starting camera thread. For cam: ' + str(cam))
        while (VideoCamera.thread_runflag[cam]):
            frames_iterator = cls.frames(cam)
            ident = get_ident()
            print(str(cam)+" Backgroundthread start: " + str(ident))
            for frame in frames_iterator:
              if not (VideoCamera.thread_runflag[cam]): break
                # and (VideoCamera.cameraState[cam]):
              if (0.9/(VideoCamera.targetfps[cam]) + VideoCamera.lasttime[cam] < time.time()):
                  # print(str(cam)+" Backgroundthread lock: " + str(ident))
                  if VideoCamera.thread_events:
                      if not VideoCamera.unlock.wait(5): print(str(cam)+" Backgroundthread locked. timout for cam: " + str(ident))
                      #print(str(cam)+" Backgroundthread unlocked and wait for request: " + str(ident))
                      if not VideoCamera.req[cam].wait(1):
                        #print(str(cam)+" Backgroundthread request. timout for cam: " + str(ident))
                        time.sleep(0)
                      VideoCamera.req[cam].clear()
                  if frame != b'':
                      VideoCamera.lastImage[cam] = frame
                      # print(str(cam)+" Backgroundthread SHOT " + str(ident))
                      #print("Frame of Camera: " +  str(cam) + " FPS:" + str('{:.0f}'.format(round(VideoCamera.fps[cam],0)))+"/"+str(VideoCamera.targetfps[cam]))
                      if VideoCamera.thread_events: VideoCamera.event[cam].set()
                      VideoCamera.setFPStime(cam, time.time())
                     # time.sleep(0)
              else:
                  #print("Call to fast for  buffer cam " + str(cam))
                   if VideoCamera.thread_events: VideoCamera.req[cam].set()
              time.sleep(0.03)
        print('Warn: STTOPPING camera thread. For cam: ' + str(cam))
          
    @classmethod
    def startprusa(cls, HTTP_URL, camera1, FINGERPRINTA, TOKENA, camera2, FINGERPRINTB, TOKENB):
       print("Start prusa service")
       if VideoCamera.prusa is None:
         VideoCamera.prusa = threading.Thread(target=cls._prusathread, daemon=False, args=(HTTP_URL, camera1, FINGERPRINTA, TOKENA, camera2, FINGERPRINTB, TOKENB))
         VideoCamera.prusa.start()
         
    @classmethod
    def stopprusa(cls):
       if VideoCamera.prusa is not None:
         print("PRUSA STOP")
         VideoCamera.prusa = None
         
    @classmethod
    def starttimelapse(cls, camera1):
       if VideoCamera.timelapse is None:
         VideoCamera.timelapse_flag=True
         VideoCamera.timelapse = threading.Thread(target=cls._timelapse, daemon=False, args=(camera1, ))
         VideoCamera.timelapse.start()
         print("timelapse  running")
       else:
         print("timelapse already running")

    @classmethod
    def renderTimelapse(cls, folder=None):
      if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "timelapse")):
        print("START RENDER")
  	    # Get the list of images in the folder.
        if folder is None:
          folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "timelapse")
          subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
          print(str(subfolders))
        else:
          subfolders=[folder]
          print("single folder" + str(subfolders))
    	    #images = os.listdir(VideoCamera.timelapse_folder)
        for subfolder in subfolders:
          print(subfolder + "---->")
          images = os.listdir(subfolder) 
    	    # Create a video file.
          images.sort()
          if len(images)==0: continue
          print(images)
          # read image
          img = cv2.imread(os.path.join(subfolder, images[0]), cv2.IMREAD_UNCHANGED)
          # get dimensions of image
          if img is not None:
            dimensions = img.shape
             
            # height, width, number of channels in image
            height = img.shape[0]
            width = img.shape[1]
            #channels = img.shape[2]
             
            print('Image Dimension    : ',dimensions)
            print('Image Height       : ',height)
            print('Image Width        : ',width)
            #print('Number of Channels : ',channels)
                 
            out_file = os.path.join(subfolder, "timelapse.avi")
            # choose codec according to format needed
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video = cv2.VideoWriter(out_file, fourcc, 12, (width, height))
            i = 0
            for image in images:
      	        image_file = os.path.join(subfolder, image)
      	        if image_file.endswith(".jpg"):
      	          frame = cv2.imread(image_file, cv2.IMREAD_COLOR)
      	          video.write(frame)
      	          #os.remove(image_file)
      	          i += 1
      	          print("Rendering frame {} / {}".format(i, len(images)))
            for image in images:
              image_file = os.path.join(subfolder, image)
              if image_file.endswith(".jpg"):
                os.remove(image_file)


    @classmethod
    def stoptimelapse(cls, camera1):
       if VideoCamera.timelapse is not None:
        # VideoCamera.timelapsestop()
         VideoCamera.timelapse_flag=False
         VideoCamera.timelapse.join(timeout=60)
         VideoCamera.timelapse = None
         # VideoCamera.renderTimelapse(folder=VideoCamera.timelapse_folder)
         print("timelapse  stopped")
       else:
         print("timelapse not running")
     
    @classmethod
    def _prusathread(cls, HTTP_URL, camera1, FINGERPRINTA, TOKENA, camera2, FINGERPRINTB, TOKENB):
        while VideoCamera.prusa is not None:
            #time.sleep(20)
            snap1=VideoCamera(camera1, "printer", 270, True).get_frame(True)
            snap2=VideoCamera(camera2, "box", 180, True).get_frame(True)
            VideoCamera.prusa_send(snap1, HTTP_URL, FINGERPRINTA, TOKENA)
            VideoCamera.prusa_send(cls, snap2, HTTP_URL, FINGERPRINTB, TOKENB)
            time.sleep(20) 
        
    @classmethod
    def _DAEMONthread(cls, cam):    
        print('Starting daemon thread. For cam: ' + str(cam))
        while VideoCamera.DAEMONthread_runflag[cam]:
            for sleep in range(45):
              time.sleep(1)
              if VideoCamera.errorFrames[cam] > 0:
                print("warn: too much error frames....resetting: " + str(cam))
                VideoCamera.errorFrames[cam] = 0
                #VideoCamera.resetcam(cam)
              if not VideoCamera.DAEMONthread_runflag[cam]: return
            lastClient = time.time() - VideoCamera.last_access[cam]
            if lastClient > 15: print("Watchdog cam: " + VideoCamera.description[cam] + "letztes Update vor: " + str(round(lastClient,2)))
            if lastClient > 65:
                # VideoCamera.thread[cam].join()
                VideoCamera.stop(cam)
                print('Stopping camera thread due to inactivity.')
                break
            #VideoCamera.thread[cam] = None
            #VideoCamera.DAEMONthread[cam] = None
            
    @classmethod        
    def _timelapse(cls, cam, time_run=22000, span=45, shots=10):
     # Create the timelapse folder in the home directory.
     if not os.path.exists("timelapse"):
         os.mkdir("timelapse")
     # Get the current date and time.
     hres_timelapse=True
     now = datetime.today()
     date = now.strftime("%Y-%m-%d")
     timehm = now.strftime("%H-%M")
     if hres_timelapse:
       width = VideoCamera.pxl_H_hres[cam]
       height = VideoCamera.pxl_V_hres[cam]
     else:
       width = VideoCamera.pxl_H[cam]
       height = VideoCamera.pxl_V[cam]
     print("width: " + str(width) + " height: " + str(height))
     # Create a folder in the timelapse folder with the current date and time.
     VideoCamera.timelapse_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "timelapse", date + "_" + timehm)
     VideoCamera.timelapse_folder_old = os.path.join(os.path.dirname(os.path.realpath(__file__)), "timelapse", date + "_" + timehm + "_old")
     out_file = os.path.join(VideoCamera.timelapse_folder, "timelapse.avi")
     out_file_old = os.path.join(VideoCamera.timelapse_folder_old, "timelapse.avi")
     # choose codec according to format needed
     print("Folder Timelapse: " + str(VideoCamera.timelapse_folder))
     if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "timelapse")):
         os.mkdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "timelapse"))
     if not os.path.exists(VideoCamera.timelapse_folder):
         os.mkdir(VideoCamera.timelapse_folder)
     if not os.path.exists(VideoCamera.timelapse_folder_old):
         os.mkdir(VideoCamera.timelapse_folder_old)
     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
     video = cv2.VideoWriter(out_file, fourcc, 12, (width, height))
     video_old = cv2.VideoWriter(out_file_old, fourcc, 12, (width, height))
     print("Timelapse started")
     #time.sleep(span)
     last_compare_image = b''
     image = b''
     best_image = image
     dif_b = image
     for loop in range(time_run):
         if not VideoCamera.timelapse_flag: break
         start_loop = time.time()
         print("Timelapse cam created")
         # Get a frame from the webcam.
         timelapseCam=VideoCamera(cam, "printer", 270, True)
         takeShots=True
         #takeShotsCount=0
         framesshot = []
         framesshot_time = []
         while takeShots:
             if not VideoCamera.timelapse_flag: break
             frame = timelapseCam.get_frame(ss=True, hres=hres_timelapse)
             framesshot.append(frame)
             framesshot_time.append(time.time())
             #takeShotsCount+=1
             if (((time.time() - start_loop) > shots) or (loop==0)) and  (frame != b''): 
               takeShots=False
             else: time.sleep(0.2)
         print("Auswertung von Photos: " + str(len(framesshot)))
         best_s = 0
         best_shot = 0
         for index, frame in enumerate(framesshot):
           if not VideoCamera.timelapse_flag: break
           #calculation_time = time.time()
           time.sleep(0.2)
           image_array = np.frombuffer(frame, np.uint8)
           image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
           gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           if(index==0):
             file_name_o = os.path.join(VideoCamera.timelapse_folder_old, "%d.jpg" % (time.time()))
             #cv2.imwrite(file_name_o, image)
             video_old.write(image)
           if (loop==0):
             print("first image of timelapse shot") 
             last_compare_image=gray_image
           #try: 
           score, diff = ssim(last_compare_image, gray_image, full=True)
           #calculation_time = time.time() - calculation_time
           #print("Shot " + str(index) + " ssim is " + str(score) + " Rechenzeit: " + str(round(calculation_time,2)))
           print("Shot " + str(index) + "/" + str(len(framesshot)) +" ssim is " + str(score))
           if(score > best_s):
             #print(str(score) + ">" + str(best_s))
             best_s = score
             best_shot = index
             best_shot_time=framesshot_time[index]
             best_image = image
             #best_frame = frame
             best_gray_image = gray_image
                # diff = (diff * 255).astype("uint8")
                # diff_box = cv2.merge([diff, diff, diff])
                # dif_b = diff_box   
           #except:
           #print("An exception occurred")
         print("Shot " + str(best_shot) + " saved") 
         file_name = os.path.join(VideoCamera.timelapse_folder, "%d.jpg" %best_shot_time)
         #cv2.imwrite(file_name, best_image)
         video.write(best_image)
         #file_name_dif = os.path.join(VideoCamera.timelapse_folder, "%d_dif.jpg" % (time.time()))
         #cv2.imwrite(file_name_dif, dif_b)         
         last_compare_image=best_gray_image
         end_loop = time.time()
         dif_loop = end_loop - start_loop
         wait_time = end_loop - (best_shot_time + span)
         if not VideoCamera.timelapse_flag: break
         print("Time of loop: "+ str(dif_loop) + "s Time to wait: " + str(wait_time) + "s")
         if wait_time>0: time.sleep(wait_time)
     print("Timelpase Ende!!!")

	
    def get_camera_settings_json():
       print("json")
       if VideoCamera.prusa is not None: prusaflag=True
       else: prusaflag=False
       camera_param = {
	        "max_cams": 3,
	        "timelapse_flag": VideoCamera.timelapse_flag,
	        "timelapse_folder": VideoCamera.timelapse_folder,
	        "prusa": prusaflag,
	    }
       return json.dumps(camera_param, indent=4)