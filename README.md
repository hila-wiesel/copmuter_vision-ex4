#  Stereo Matching,  Homography and Warping
### copmuter_vision_ex4

in this test I implements 4 thing:

## 1. disparity image -
function which takes two images, Left and Right, and finds outputs the disparity map

the images:
## left: , right:
![pair0-R](https://user-images.githubusercontent.com/61710157/123822427-f0eb9680-d904-11eb-9011-2148a11238ee.png) ![pair0-L](https://user-images.githubusercontent.com/61710157/123822392-e92bf200-d904-11eb-96e4-71b83f3bb471.png)

 
## the results:
![Depth Image by SSD](https://user-images.githubusercontent.com/61710157/123823000-71aa9280-d905-11eb-8cec-bc34255bff99.png)
![Depth Image by NCC](https://user-images.githubusercontent.com/61710157/123823015-75d6b000-d905-11eb-9979-397a17eb06e2.png)


## 2. homogrpy-
a function which takes four, or more, pairs of matching keypoints and returns the homography
matrix H.


## 3. warping-
function which takes two images (eg. a poster and a billboard), and warps the poster on to the
billboard. The user will mark the source points and their matches on the destination image

the two images:
![billBoard](https://user-images.githubusercontent.com/61710157/123822488-0365d000-d905-11eb-92c5-d9d0d08b429d.jpg)
![car](https://user-images.githubusercontent.com/61710157/123822508-082a8400-d905-11eb-88f8-8716da46fb48.jpg)


the results:
![Warping](https://user-images.githubusercontent.com/61710157/123822469-fea11c00-d904-11eb-8896-699932b36d84.png)

