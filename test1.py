import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()  # separates background from foreground 

#-------------------------------------------------------------------------------------------------------------
##                                      BASIC INITIALIZATION OF THE RADAR CAM  
#-------------------------------------------------------------------------------------------------------------

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Video access failed")
        break
    else:
    # 1) applying grayscaling on the feed , implementing background and foreground separator using fgbg 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts colors into gray color for easier model learning 
        frame = cv2.resize(frame, (640, 480))           # resizing frame 
        fgmask = fgbg.apply(frame)                      # applying masking -> detects the motion in the frames


    # 2)  removing noise from the mask so noise can be removed for the machine to learn the env more efficiently 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))   
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)       # morphology -> removes the noise 
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)


    # 3) using contours to detect shapes from white blobs on the radar feed + filtering out small noise particles

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:        ## tuning parameter. filtering by area. keeping 500 as baseline to allow better detection
                continue


    # 4) Drawing boxes onto the clean mask for better detection and clarity 
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                            #starting , #end point ,# colorGBR #thickness

                            

    ##----------------------------------------------------------------------------------------------------------##
    #                  COMPARING POSITIONS ACROSS FRAMES FOR MODEL TO REMEMBER PREVIOUS FRAME 
    ##----------------------------------------------------------------------------------------------------------##


    # starting  the radar camera 
        cv2.imshow("Motion Mask", fgmask)
        print(frame.shape)
    

    
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()