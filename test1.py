import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()  # separates background from foreground 
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # apply the Background mask 

    if not ret:
        print("Video access failed")
        break
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (640, 480))
        fgmask = fgbg.apply(frame)
        cv2.imshow("Motion Mask", fgmask)
        print(frame.shape)
    

    
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()