## **Vehicle Detection Project**


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[project_output]: ./projec_output.mp4
[histogram]: ./writeup_images/hist-compare.jpg
[spatial-binning]: ./writeup_images/spatial-binning.jpg
[car-and-hog]: ./writeup_images/car-and-hog.jpg
[boxes]: ./writeup_images/boxes.png
[cars_found]: ./writeup_images/cars_found.png
[heatmap]: ./writeup_images/heatmap.png
[histogram2]: ./writeup_images/histogram.png
[rgb-histogram-plot]: ./writeup_images/rgb-histogram-plot.jpg
[scalar_fit]: ./writeup_images/scalar_fit.png
[single_boxes]: ./writeup_images/single_boxes.png
[sliding-window]: ./writeup_images/sliding-window.jpg
[multiple_windows]: ./writeup_images/multiple_windows.png

## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view) 

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

## Writeup / README

First, to find Cars in images, I got information related to color, saturation, and luminosity for each car and non car example extracting a histogram and use it to create a feature vector. Here is a example: 

![alt text][histogram]

After that, Define a function to compute color features, given a color_space flag as 3-letter all caps string
like 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'. In this function it's possible define a new size to reduce the total amount of features without lose relevant informations. Here is a example:

![alt text][spatial-binning]


## Importing Data

To feed our algorithm we have two option of dataset: a Small used to test and tune the algorithm, and a full to create a more robust classifier.

Data can be downloaded at,

**Small Dataset:** 

[Vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip)
[Non-Vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles_smallset.zip)

**Full Dataset:**

[Vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
[Non-Vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)


In final version I used the full datase to train algorithm that contains: 

 8792  examples of Cars 
 8968  examples of Non Cars
 
 ![alt text][image1]
 

### Histogram of Oriented Gradients (HOG)

Color Histograms do not capture information about shape of objetcs, so I compute a [Histogram of Oriented Gradient](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html) to extract this information and like I did with Color Histogram use it as feature to feed our classifier.


```python

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Extract features and Hog Image
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        # Extract features and Hog Image
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features
 ```
 Here a example of HOG:
 
 ![alt_text][car-and-hog]
 
 ## Combine Features
 
 To feed our classifier we need the most possible number of feature, in this project we'll use 3 types:

* **Spatial Binning of Color:** which allow us perform a template match. This is not the best way because don't it takes into account changes of shape or variations, but still usefull to improve our algorithm.


* **Color Histogram :** Capture signature of color, saturation and luminosity of car and non car examples.


* **Histogram of Oriented Gradient:** Capture information about shape of car and non car examples.



 Then I define a function to search cars given a image and a list of windowns, 
 
 ```python
 
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

```

## Building a Classifier

To build a classifier, I follow the steps bellow:

* First define parameters
* Extratct Features
* Normalize Features
* Create a Train Dataset with Features (X) and Labels (y)
* Shufle the data
* Train a Classifier
* Score Classifier Performance

---
1. Parameter used:

```python
#Tweak these parameters and see how the results change.
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [500, None] # Min and max in y to search in slide_window()
```

2. Extratct Features

```python
# Extract Features
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat) 

```

3. Normalize Features

```python
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

```

4. Create a Train Dataset with Features (X) and Labels (y)

```python
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

```

5. Shufle the data

```python
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

```


6. Train a Classifier

To this project Suppor Vector Machine works well, I used [GridSearCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to find given a list of parameters the best combination, use [Cross Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) to score then. 

```python
# Use a linear SVC 
parameters = {'kernel':['linear'], 'C':[0.1, 0.2, 0.3, 0.5, 1]}
svr = SVC()
svc = GridSearchCV(svr, parameters)

t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

```

3248.62 Seconds to train SVC...
Test Accuracy of SVC =  0.9862
Support Vector Machine Best Parameters: 
Kernel= linear , and C= 0.1

7. Score Classifier Performance

```python
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

```
 Test Accuracy of SVC =  0.9862
  
## Sliding Windows

To find cars in each image or video frame, Iǘe implemented a function to extract small portions (windows) of original image, using given parameters we'll able to define how windows will "sliding" through image, position, overlay and size.

```python

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


```
Example of result using Sliding Windows function:

![alt text][cars_found]

Parameters used:

* **y start point:** 400
* **y stop point:** 400
* **scale of windows:** 1.5

Then I defined a function to extract features using hog sub-sampling and make predictions

```python

def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    bbox_list=[]
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                bbox_list.append(((xbox_left, ytop_draw+ystart), #github.com/preritj
                                  (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    
    return draw_img, bbox_list
```

## Defining a mutiple_scale_find_cars

Defining a optimized version of find_cars function, that get a list of parameters as argument and solve "perspective problem", in other words, cars appears smaller when they is far, and bigger when is near. So using a list of parameters our algorithm find cars using proper windows sizes in each vertical lane.

![alt text][multiple_windows]

```python

# Define a function to find cars in multiple windows sizes
def mutiple_scale_find_cars(img, list_ystart, list_ystop, list_scale):
    
    box_list = []
    
    # For each parameter in parameter lists (list_ystart, list_ystop, list_scale)
    for start, stop, scale in zip( list_ystart, list_ystop, list_scale):
    
        out_img, box_temp = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        
        # If car was found feed box list
        if (len(box_temp) != 0):
            box_list += box_temp
    
    return box_list
```

### Result of Multiple Scale Find Cars

I've tested many parameters combinations, for the project I used the following parameters:

| y start | y stop | window scale |
|:-------:|:------:|:------------:|
|   400   |   470  |    1.0       |
|   410   |   480  |    1.0       |
|   400   |   500  |    1.5       |
|   430   |   530  |    1.5       |
|   400   |   530  |    2.0       |
|   430   |   560  |    2.0       |
|   400   |   600  |    3.5       |
|   460   |   700  |    3.5       |

![alt text][boxes]

## Draw a Single Box and Heatmap

As we've seen in results above, our function draw many boxes if car were found more than one time, we can use that information to create a more robust method that can be able to exclude false positives and where a car were found with por confidence draw a unique box.

To indentify the number and positions of cars found in heatmap we'll use [Scipy Label Function](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html) that will give and this information.

Function to add "heat" in a heatmap where box were found

```python

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

```
Function to clean areas where possible false positives were found

```python

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
    
```
Function to Draw a single box, using scipy label function

```python

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
```

Here a example of a Heatmap:

![alt text][heatmap]

## Pipeline (put all peaces together)

To indentify car in each image we need to perform all steps described above:

1. Read a image
2. Find Cars using multiple scale windows and get a Boxes List where they were found
3. Using a Boxes List add heat in a heatmap
4. Apply a Threshold to clean false positives
5. Using Scipy Label function "locate and count" cars found
6. Draw a single bounding box for each car found

To run the pipeline above I defined the following function:

```python
def process_image(img):
    
    # Heat image like img
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    
    # Get a boxes list
    box_list = mutiple_scale_find_cars(img, list_ystart, list_ystop, list_scale)
    
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    return draw_img  
    
```
Result of process_image function

![alt text][single_boxes]

## Improving Car Detection

In project video using process_image function I get acceptable results, but with many "flickering" between video frames. So, 
to improve this result and create a better and smoother detection system I used information from previous frame to analize actual frame image, this way our result will be a lot smoother and eventual false posives that apper in only one frame will be removed.

Define a class to store data from video:

```python
class Vehicle_Detect():
    def __init__(self):
        # history of rectangles previous n frames
        self.prev_rects = [] 
        
    def add_rects(self, rects):
        self.prev_rects.append(rects)
        if len(self.prev_rects) > 15:
            # throw out oldest rectangle set(s)
            self.prev_rects = self.prev_rects[len(self.prev_rects)-15:]
 ```
* Code for Class to store data from video was copied from [Je'remy Shannon's work](https://github.com/jeremy-shannon/CarND-Vehicle-Detection/blob/master/vehicle_detection_project.ipynb)

Define a process frame for video:

```python

def process_frame_for_video(img):
    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    
    # Get boxes list    
    box_list = mutiple_scale_find_cars(img, list_ystart, list_ystop, list_scale)
    
    # add detections to the history
    if len(box_list) > 0:
        det.add_rects(box_list)
    
    heatmap_img = np.zeros_like(img[:,:,0])
    for rect_set in det.prev_rects:
        heatmap_img = add_heat(heatmap_img, rect_set)
    heatmap_img = apply_threshold(heatmap_img, 1 + len(det.prev_rects)//2)
     
    labels = label(heatmap_img)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img
```
## Final Output

We can see that when consider information from previous frame we get better and smoother results.

[Project Video on Youtube](https://youtu.be/Qh8H45vjTOA)


[link to my video result](./project_output.mp4)


---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach worked well but I've noticed some false positives in project video, one way to avoid this is optmize the classifier using more examples and trying other algorithms and parameters. 

The parameters used in Sliding Windows was defined with experimentation, a way to improving it could be using math and geometry to understand how (and how much variation) cars size changes with distance. 

In improved algorithm I consider only **one** previous frame to create heatmap to actual image, use more frames can be a way to create a more robust algorithm, with more detection (more precise bounding box) and less false positives. 


