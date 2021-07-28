import face_recognition as fr
from PIL import Image, ImageDraw
import os
import datetime

encoded_images = []
img_labels = []

for filename in os.listdir("dataset"):
    print(filename)
    user_image = fr.load_image_file('./dataset/'+filename)
    encoded_image = fr.face_encodings(user_image)[0]
    # print(encoded_image)
    encoded_images.append(encoded_image)
    img_labels.append(filename)

# Load test image to find faces in
test_image = fr.load_image_file('./test/angg-11.jpg')

start_time_all = datetime.datetime.now()
# Find faces in test image
face_locations = fr.face_locations(test_image)
face_encodings = fr.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = fr.compare_faces(encoded_images, face_encoding,tolerance=0.4)

  name = "Unknown Person"

  # If match
  if True in matches:
    first_match_index = matches.index(True)
    name = img_labels[first_match_index]
  
  print(matches)
  # Draw box
  draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

  # Draw label
  text_width, text_height = draw.textsize(name)
  draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
  draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

del draw

end_time_all = datetime.datetime.now()
time_diff_all = (end_time_all - start_time_all)
execution_time = time_diff_all.total_seconds()
print(execution_time)

# Display image
pil_image.show()

