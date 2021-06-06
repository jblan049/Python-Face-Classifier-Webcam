import cv2 #Import OpenCV

cap = cv2.VideoCapture(0) #Capture video from webcam

# Check if the webcam is open correctly
if not cap.isOpened():
	raise IOError("Cannot open webcam")
	
while True: #Infinite Loop
	ret, frame = cap.read() #Read a frame from the webcam
	frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) #Resize the image
	faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #Create a classifier based on the inputted xml file
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Create a grayscale image of the frame
	
	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
    	gray,
    	scaleFactor=1.1,
    	minNeighbors=5,
    	minSize=(30, 30),
    	flags = cv2.CASCADE_SCALE_IMAGE
	)	
	
	#Draw the bounding box around the face
	for (x, y, w, h) in faces:
    		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
	
	cv2.imshow('Face Detection',frame) #Show the frame with the bounding box
	
	c = cv2.waitKey(1) #Read keyboard input
	if c == 27: #If keyboard input was 'Esc'
		break #Break out of loop
		
cap.release() #Release the cap variable from memory
cv2.destroyAllWindows() #Destroy all OpenCV windows
