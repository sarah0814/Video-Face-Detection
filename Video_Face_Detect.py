
# coding: utf-8

# In[14]:


import cv2
import numpy as np
import requests
from json import JSONDecoder
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import time

http_url='https://api-cn.faceplusplus.com/facepp/v3/detect'
key = "Dc9H5NAqeNDg4Pa_cdBu8c8fktEfm0FP"    
secret = "Ol7IuZhGpBKjLeseK344CyvZs1jo_tvl"
attr = "gender,age,smiling,emotion,ethnicity,beauty"
data = {
        'api_key': key,
        'api_secret': secret,
        'return_attributes': attr
}

def face(image_path):
    for i in range(20):
        try:
            response = requests.post(http_url, data=data, files={"image_file": open(image_path, "rb")})
            req_con = response.content.decode('utf-8')
            req_dict = JSONDecoder().decode(req_con)
            #print("%d faces are detected" %len(req_dict['faces']))
            if "error_message" in req_dict.keys():
                time.sleep(1)
                continue
            else:
                return((req_dict))
        except requests.exceptions.HTTPError as e:
            print(e)
        if i==19:
            print(None)

def showface(result, image_path, image_labeled_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("./abc.ttf", 40)
    faces = result["faces"]
    faceNum = len(faces)
    if faceNum < 1:
        img.save(image_labeled_path)
    else:
        for i in range(faceNum):
            face_rectangle = faces[i]['face_rectangle']
            width =  face_rectangle['width']
            top =  face_rectangle['top']
            left =  face_rectangle['left']
            height =  face_rectangle['height']
            #start = (left, top)
            #end = (left+width, top+height)
            color = (55,255,155)
            thickness = 3
            if str(result["faces"][i]["attributes"]["gender"]["value"]) == 'Female':
                draw.rectangle((left, top, left+width, top+height), outline = "red")
                draw.text((left-200, top-50), str(result["faces"][i]["attributes"]["age"]["value"]), fill = 'red', font=font)
                draw.text((left-130, top-50), str(result["faces"][i]["attributes"]["gender"]["value"]), fill = 'red', font=font)
                draw.text((left-20, top-50), max(result["faces"][i]["attributes"]["emotion"], key=result["faces"][i]["attributes"]["emotion"].get), fill = 'red', font=font)
            else:
                draw.rectangle((left, top, left+width, top+height), outline = "yellow")
                draw.text((left-100, top-50), str(result["faces"][i]["attributes"]["age"]["value"]), fill = 'yellow', font=font)
                draw.text((left-40, top-50), str(result["faces"][i]["attributes"]["gender"]["value"]), fill = 'yellow', font=font)
                draw.text((left+40, top-50), max(result["faces"][i]["attributes"]["emotion"], key=result["faces"][i]["attributes"]["emotion"].get), fill = 'yellow', font=font)

        img.save(image_labeled_path)

def videoFaceDetect(in_video_path, out_video_path, image_path, image_labeled_path):
    cap = cv2.VideoCapture(in_video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    print("frame rate: " + str(frame_rate) + "; frame width: " + str(frame_width) + "; frame height: " + str(frame_height))

    vidout = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width,frame_height))                 
    
    count = 0
    success = True
    
    while success:
        count = count + 1
        success,img = cap.read()
        if success:
            cv2.imwrite(image_path, img)
            if count%100 == 0:
                print('Complete frame #{}: {}'.format(count, success))
            result = face(image_path)
            showface(result, image_path, image_labeled_path)
            img_labeled = cv2.imread(image_labeled_path)
            vidout.write(img_labeled)
    vidout.release()


# In[ ]:


videoFaceDetect('./HappyBirthday_2K.mp4', '/var/Sarah/HappyBirthday_labeled.mp4', './Temp/temp_image.jpg', './Temp/temp_labeled_image.jpg')


# In[ ]:




