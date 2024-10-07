import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import cv2

# Initialize Firebase Admin SDK
cred = credentials.Certificate("ServiceAccount.json")
firebase_admin.initialize_app(cred, {"storageBucket": "spectrophotometer-97c4f.appspot.com"})

# Access the storage bucket
bucket = storage.bucket()

# Get the image blob from Firebase Storage
blob = bucket.get_blob("SentImage1.jpg")

# Download the image as a string and convert it to a numpy array
arr = np.frombuffer(blob.download_as_string(), np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)


cv2.imshow("image", img)
cv2.imwrite('saved_image.jpg',img)
print("image save to saved_image")
cv2.waitKey(0)
cv2.destroyAllWindows()