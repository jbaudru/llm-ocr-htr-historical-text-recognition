import cv2
import numpy as np

class Image:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
    def color_image(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, self.image = cv2.threshold(self.image, 120, 255, cv2.THRESH_BINARY)
        return self.image

    def save(self, output_path):
        cv2.imwrite(output_path, self.image)
        
    def crop_image(self):
        #user32 = ctypes.windll.user32
        #screensize = user32.GetSystemMetrics(0)//2, user32.GetSystemMetrics(1)//2
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
        blurred = cv2.GaussianBlur(gray, (35, 35), 0) # Apply Gaussian blur to the image to reduce noise
        alpha = 1.9 # Contrast control (1.0-3.0)
        beta = 20 # Brightness control (0-100)
        contrasted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
        edges = cv2.Canny(contrasted, 29, 29) # Use Canny edge detection to find edges in the image
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 500, minLineLength=2000, maxLineGap=500) # Use Hough Line Transform to find lines in the image
        min_x, min_y = self.image.shape[1], self.image.shape[0]
        max_x, max_y = 0, 0
        try:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Update the smallest and largest x and y values
                min_x, min_y = min(min_x, x1, x2), min(min_y, y1, y2)
                max_x, max_y = max(max_x, x1, x2), max(max_y, y1, y2)
            # Crop the image vertically and horizontally
            cropped = self.image[min_y:max_y, min_x:max_x]
            """
            image_tmp = cv2.resize(image, screensize)
            cv2.imshow('Image with Rectangle', image_tmp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
            #print("[INFO] Crop done!")
            return cropped
        except:
            #print("[INFO] No crop possible!")
            return self.image