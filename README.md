# Harris-corner-detection
corner detections in image using an implementation of Harris corner detection algorithm

## how it works
-	Convert the input image to gray scale
-	Compute the gradients of the image
-	Compute the second moment matrix M
-	Apply Guassian filter to M
-	Compute the Haris mesure or the Harmonic Mean
-	Segmentation on the Cornerness Haris measure or Harmonic Mean 
-	search local maximum from the 8 neighbors

<div>
<img src="https://raw.githubusercontent.com/nassim-fox/Harris-corner-detection/master/A.jpg" width="300">
<img src="https://raw.githubusercontent.com/nassim-fox/Harris-corner-detection/master/s.PNG" width="300">
  </div>
