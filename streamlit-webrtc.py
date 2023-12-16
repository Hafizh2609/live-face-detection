from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import cv2, os
from twilio.rest import Client
account_sid = "AC2d41578652fa01388e8df7cde1cdb17b"
auth_token = "beebe4092916f445a5bddf56b6682b65"
client = Client(account_sid, auth_token)
token = client.tokens.create()

model_face = YOLO('yolov8n-face.pt')
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    flip_frame=cv2.flip(img, 1)
    output_face=model_face.predict(flip_frame)
    for obj in output_face[0].boxes:
        # Get bounding box coordinates
        x1,y1,x2,y2 = obj.xyxy[0]

        # Get class label
        class_id = model_face.names[int(obj.cls)]
        cv2.rectangle(flip_frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),5) # green color
        cv2.putText(flip_frame, class_id, (int(x1), int(y1)-10 if y1-10>0 else 0), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(flip_frame, format="bgr24")

webrtc_streamer(key="example", video_frame_callback=video_frame_callback,rtc_configuration={"iceServers": token.ice_servers})
