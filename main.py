import cv2
import time
import imutils
from playsound import playsound
import datetime


cap = cv2.VideoCapture(0)
frame_count = 0
time.sleep(2)

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
# Cascades require a greyscale image to do the classification.

#background_subtr_method = bgs.SuBSENSE()

# while True:
#     # read video frames
#     retval, frame = cap.read()

#     # check whether the frames have been grabbed
#     if not retval:
#         break

#     # resize video frames
#     frame = cv2.resize(frame, (640, 360))

#     # pass the frame to the background subtractor
#     foreground_mask = background_subtr_method.apply(frame)
#     # obtain the background without foreground mask
#     img_bgmodel = background_subtr_method.getBackgroundModel()

#      # show the current frame, foreground mask, subtracted result
#     cv2.imshow("Initial Frames", frame)
#     cv2.imshow("Foreground Masks", foreground_mask)
#     cv2.imshow("Subtraction result", img_bgmodel)

#     keyboard = cv2.waitKey(1)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

firstFrame = None


while True:
    _, frame = cap.read()

    text = "No motion"

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        continue

    frameDelta = cv2.absdiff(firstFrame, gray)
    
    thresh = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
	
    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) < 3000:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #playsound('C:\\Users\\danie\\Documents\\GitHub\\detect-a-cat\\audio.wav')
        text = "Occupied"

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.putText(frame, str(datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")) + "    " + str(frame_count), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)

    frame_count += 1

    if frame_count > 180:
        firstFrame = gray
        frame_count = 0

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()