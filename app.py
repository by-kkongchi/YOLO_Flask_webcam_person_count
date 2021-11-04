from logging import Formatter
from flask import Flask, jsonify,request,Response
from yolo_detection_webcam import detectObjects
import cv2
app = Flask(__name__)


video = cv2.VideoCapture(0)

@app.route('/')
def gen(video):
    while True:
        success, image = video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
 
@app.route('/camera')
def video_feed():
    global video
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame') 

@app.route('/myapp/detectObjects')
def detect():
    #frame = request.args['frame']

   video_path = Response(gen(video)) 
   results = detectObjects(video_path)
    
   return jsonify(results)



app.run()