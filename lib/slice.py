import cv2
import os
# Paths
path = os.path.dirname(os.getcwd())
input_folder = path+"/data/lines"
output_folder = path+"/data/sliced_words"

def process_image(image_path, output_folder):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to create a binary image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply dilation to connect letters within words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))  # Adjust kernel size as needed
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each contour
    for i, contour in enumerate(contours):
        # Get bounding box for each word
        x, y, w, h = cv2.boundingRect(contour)
        
        # Ignore small contours (noise)
        if w > 30 and h > 10:  # Adjust thresholds as needed
            word_image = image[y:y+h, x:x+w]  # Crop the word
            output_path = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_word_{i+1}.jpg")
            cv2.imwrite(output_path, word_image)
            print(f"Saved: {output_path}")


# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all image files from the input folder
input_images = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(('.jpg'))]

# Process each image
for image_path in input_images:
    process_image(image_path, output_folder)
