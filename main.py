#!/usr/bin/env python

from flask import Flask, render_template, Response, make_response, request, jsonify
from camera import VideoCamera  # Import the VideoCamera class from the camera module
import time
import pyaudio
import json
from dotenv import load_dotenv
import os

# load .env-Datei
load_dotenv()

# read as global variables
FINGERPRINT = os.getenv("FINGERPRINT")
TOKEN = os.getenv("TOKEN")
camera1_name = os.getenv("camera1_name")
camera2_name = os.getenv("camera2_name")

print(FINGERPRINT)
print(TOKEN)

# Pre-source definition if auto-detect is wrong 
camera1 = 0
camera2 = 1

# Automatic detection 
camera1_name = "HD Pro Webcam C920"
camera2_name = "camera0"

# Initialize the Flask app
app = Flask(__name__)
fps = 0

# Render the index page
@app.route('/')
def index():
    return render_template('index.html')

# Generator function for streaming video frames
def gen(camera):
    global fps
    while True:
        frame = camera.get_frame()
        if not frame == None:
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Endpoint for serving combined image from two cameras
@app.route('/image')
def image():
    snap1 = VideoCamera(camera1, "printer", 270, True).get_frame(True)
    snap2 = VideoCamera(camera2, "box", 180, True).get_frame(True)
    snapsum = VideoCamera.sum_frame(snap1, snap2)  # Calling sum_frame() to get the combined image
    mimetype = 'image/jpeg'
    return Response(snapsum, mimetype=mimetype)
    
@app.route('/video_feedA.mjpeg')
def video_feedA():
    global genReturnA
    genReturnA=gen(VideoCamera(camera1, "printer", 270, False))
    if not genReturnA==None:
        return Response(genReturnA, mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shotA')
def shotA():
    snap1=VideoCamera(camera1, "printer", 270, True).get_frame(True, True)
    mimetype = 'image/jpeg'
    return Response(snap1, mimetype=mimetype)

@app.route('/shotB')
def shotB():
    snap2=VideoCamera(camera2, "box", 180, True).get_frame(True, True)
    mimetype = 'image/jpeg'
    # Rückgabe der Bildantwort
    return Response(snap2, mimetype=mimetype)

@app.route('/video_feedB.mjpeg')
def video_feedB():
    global genReturnB
    genReturnB=gen(VideoCamera(camera2, "box", 180, False))
    if not genReturnB==None:
        return Response(genReturnB, mimetype='multipart/x-mixed-replace; boundary=frame')
        
@app.route('/command', methods=['POST'])
def handle_command():
    data = request.get_json()
    command = data.get('command')
    if command == 'startStreamingA':
        print('Streaming A gestartet')
        VideoCamera.setFPS(camera1, 20)
        VideoCamera.setFPS(camera2, 5)
    elif command == 'stopStreamingA':
        VideoCamera.setFPS(camera1, 20)
        VideoCamera.setFPS(camera2, 5)
        print('Streaming A gestoppt')
    elif command == 'startStreamingB':
        print('Streaming B gestartet')
        VideoCamera.setFPS(camera1, 5)
        VideoCamera.setFPS(camera2, 20)
    elif command == 'stopStreamingB':
        VideoCamera.setFPS(camera1, 20)
        VideoCamera.setFPS(camera2, 5)
        print('Streaming B gestoppt')
    elif command == 'startTimelapse':
        VideoCamera.starttimelapse(camera1)
    elif command == 'stopTimelapse':
        VideoCamera.stoptimelapse(camera1)
    elif command == 'renderTimelapse':
        VideoCamera.renderTimelapse()
    elif command == 'stopprusa':
        VideoCamera.stopprusa()
    elif command == 'startprusa':
        print("Start prusa service")
        VideoCamera.startprusa(camera1, camera2)
    elif command == 'resetcam':
        print("rest cam")
        VideoCamera.resetcam(camera1, camera2)
    elif command == 'test':
        print("test")
        VideoCamera.errorFrames[camera1] = 48
    else:
        # Unerwarteter Befehl
        print('Ungültiger Befehl: ' + command)
    # Sende eine Antwort zurück (optional)
    return 'Befehl empfangen'
    
@app.route('/json/<cmd>', methods=['GET', 'POST'])
def add_message(cmd):
    json = request.json
    content=json
    print(cmd)
    print(content)
    if content: VideoCamera.startprusa(camera1, camera2)
    elif not content=="False": VideoCamera.stopprusa()
    return jsonify({"cmd":cmd})
    
@app.route('/json')
def display_json():
    # Get the camera settings JSON using the function
    camera_settings_json = VideoCamera.get_camera_settings_json()
    camera_settings_obj = json.loads(camera_settings_json)
    # Return the JSON response
    return jsonify(camera_settings_obj)
              
if __name__ == '__main__':
    camera1=VideoCamera.get_camera(camera1_name)
    camera2=VideoCamera.get_camera(camera2_name)
    #cameras=(VideoCamera.list_ports()[1])
    print("Kamera1: " + str(camera1) + " Kamera2: " +str(camera2))
    print("Start prusa service")
    VideoCamera.startprusa(camera1, camera2)
    print("Computing...")
    VideoCamera.renderTimelapse()
    ##Timelapse Autostart
    #print("Start Timelapse")
    #VideoCamera.starttimelapse(camera1)
    print("Start Webservice")
    app.run(host='0.0.0.0', threaded=True)