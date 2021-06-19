import cv2

# we use pre-trained cascade files [face_cascade,eye_cascade,smile_cascade]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


# fun for detect face, eye and smile
def detect(gray, frame):
	
	# we will use gray_pic for detection
	faces = face_cascade.detectMultiScale(gray, 1.2, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
		
		roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
		roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
		
		# we want to detect eyes and smile within face
		eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 5)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)
			
		smile = smile_cascade.detectMultiScale(roi_gray,scaleFactor=2, minNeighbors=25)
		for (sx,sy,sw,sh) in smile:
			cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),3)
	
	return frame
	
	
# take the capture 
cap = cv2.VideoCapture(0)

while True:
	
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # we change color to gray
	
	result = detect(gray, frame)
		
	cv2.imshow('pic',result)	
	
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

cap.release()
cv2.destroyAllWindows()