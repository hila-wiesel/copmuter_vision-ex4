#  Stereo Matching,  Homography and Warping
#### copmuter vision ex4

in this test I implements 4 thing:

## 1. disparity image -
function which takes two images, Left and Right, and finds outputs the disparity map

the images:
### left: , right:
![pair0-R](https://github.com/hila-wiesel/copmuter_vision-ex4/blob/main/pair0-R.png) ![pair0-L](https://github.com/hila-wiesel/copmuter_vision-ex4/blob/main/pair0-L.png)

 
### the results:
![Depth Image by SSD](https://github.com/hila-wiesel/copmuter_vision-ex4/blob/main/Depth%20Image%20by%20SSD.png)
![Depth Image by NCC](https://github.com/hila-wiesel/copmuter_vision-ex4/blob/main/Depth%20Image%20by%20NCC.png)


## 2. homogrpy-
a function which takes four, or more, pairs of matching keypoints and returns the homography
matrix H.


## 3. warping-
function which takes two images (eg. a poster and a billboard), and warps the poster on to the
billboard. The user will mark the source points and their matches on the destination image

### the two images:
![billBoard])https://github.com/hila-wiesel/copmuter_vision-ex4/blob/main/billBoard.jpg)
![car](https://github.com/hila-wiesel/copmuter_vision-ex4/blob/main/car.jpg)


### the results:
![Warping](https://github.com/hila-wiesel/copmuter_vision-ex4/blob/main/Warping.png)

