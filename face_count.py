import cv2

# we use pre-trained cascade files [face_cascade,eye_cascade,smile_cascade]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# fun for detect face, eye and smile
def detect(gray, frame):
	
	# we will use gray_pic for detection
	faces = face_cascade.detectMultiScale(gray, 1.2, 5)
	num = len(faces)
	print(len(faces))
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
		cv2.putText(frame, 'face'+str(num),(x-12,y-12),cv2.FONT_HERSHEY_PLAIN,0.7,(0,255,0),2)
	
	return frame
	
######################################################## Capture the live video ######################################################
# take the capture 
# cap = cv2.VideoCapture(0)

# while True:
	
# 	ret,frame = cap.read()
# 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # we change color to gray
	
# 	result = detect(gray, frame)
		
# 	cv2.imshow('pic',result)	
	
# 	if cv2.waitKey(1) & 0xFF == ord('q'): 
# 		break

# cap.release()
# cv2.destroyAllWindows()

########################################################################################################################################

img = cv2.imread("1.JPG")
img = cv2.resize(img, (420,520))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

result = detect(gray_img, img)

cv2.imshow("image", result)
cv2.waitKey()

########################################################################################################################################