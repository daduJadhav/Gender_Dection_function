import argparse         #argurnment parser
import cv2              #recording video , control of camera
import math             

# highliting face
def highlightingface(net, frame, con_trasholed = 0.7):                        #con_trasholed = confidence_trasholed
    frameOpencvDnn = frame.copy()                           #Taking copy to make a 
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn,1.0,(300,300), [104,117,123],True,False)

    net.setInput(blob)
    detections = net.forward()
    frameBoxes = []
    for i in range(detections.shape[0]):
        confidence = detections[0,0,i,2]
        if confidence>con_trasholed:
            x1 = int(detections[0,0,i,2]*frameWidth)
            y1 = int(detections[0,0,i,2]*frameHeight)
            x2 = int(detections[0,0,i,2]*frameWidth)
            y2 = int(detections[0,0,i,2]*frameHeight)
            frameBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn,(x1,y1), (x2,y2), (0,255,0), int (round(frameHeight/150)), 8)
    return frameOpencvDnn, frameBoxes


parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"  
ageModel = "age_net.caffemodel"
ageProto = "age_deploy.prototxt"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

age_list = ['(1-3)', '(4-6)', '(8-12)', '(15-20)','(25-35)', '(40-60)', '(65-80)']
gender_list = ['Male','Female'] 
model_mean_value = (78.4263377603, 87.7689143744, 114.895847746)

FaceNet = cv2.dnn.readNet(faceModel,faceProto)
ageNet = cv2.dnn.readNet(ageModel,ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)



video = cv2.VideoCapture(args.image if args.image else 0)           #agar image ha to image ko lo varna camera use karo
padding = 20
while cv2.waitKeyEx(1)<0 :
    hasframe,frame = video.read()
    if not hasframe:
        cv2.waitKeyEx()
        break
    resultingImg, faceBoxs = highlightingface(FaceNet,frame)
    if not faceBoxs:
        print('Dectetion Fail.....')

    for facebox in faceBoxs:
        face = frame[max(0,facebox[1]-padding):min(facebox[3]+padding, frame.shape[0]-1),
                     max(0,facebox[0]-padding):min(facebox[2]+padding, frame.shape[1]-1)]
        
        blob = cv2.dnn.blobFromImage(face,1.0, (227,227), model_mean_value, swapRB = False)

        #Detection of gender : 
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = gender_list[genderPreds[0].argmax()]
        print(f"Gender : {gender} ")

        # Detection Of age 
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = age_list[agePreds[0].argmax()]
        print(f"Age : {age[1:-1]} ")

        cv2.putText(resultingImg, f"{gender}, {age}", (facebox[0], facebox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender",resultingImg)
