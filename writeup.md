# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

## Reflection

## 1. Build a Lane Finding Pipeline. 

My pipeline consisted of 5 steps:

* Image conversion to grayscale
* Noise removal by Gaussian smoothing
* Canny Edge Detection to detect object edges 
* Region of interest selection, where we consider only a part of an image and removing everything else in the image
* Hough Transform to detect lines

## 2. Modified the draw_lines() function.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function as follows:

* separated line segments delivered by the Hough transform into 2 **sets**, i.e. those 
  that have positive slope vs the ones with negative slope (left and right lanes)
* implemented **least squares regression method** to interpolate/extrapolate the lines 
* performed **outliers filtering**, i.e. line segments that do not belong neither to the left nor to the right lane lines: 
  - median filtering of slopes within a set (rather than just averaging all slopes within a set) that
    would deliver a slope **b** of the interpolated/extrapolated line
  - remove outlier samples from the calculation of the lane line itself, i.e. another filtering based on phase comparison, i.e. 
    **abs(slopes_of_all_segments_in_a_set - b) < Thr**, where **Thr** is some manually tuned threshold
    
To futher eleminate outliers effect on the final plot, I tuned Hough Tranform threshold paramter to allow to
collect enough line segments/statistics for the least squares regression method while still removing some portion of
line segments that are unlikely to belong to either the left or the right lanes.


## 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be that the region of interest is hard-coded to fit the test videos.
Another shortcoming could be that parametrization is quite tedios. 
Will not work well in the curves.


## 3. Suggest possible improvements to your pipeline

Compute region of interest adaptivly to the scene.
Try to use other colormaps and/or color extraction techniques in order to reduce tight parametrization constraints.
Additional filtering over time (i.e. video frames) my take in place for getting better (less "jumpy") lane lines, in fact, I already tested it but did not included it into the notebook (however videos available in the 'test_videos_output' folder)
