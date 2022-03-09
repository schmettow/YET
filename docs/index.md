---
title: "Your Eye Tracker developers guide"
author: "M Schmettow"
date: "30/09/2021"
output: html_document
---

# YET is your eye-tracker (YET)

YET is Your Eye Tracker. The goal of this project is to make eye trackers so cheap that everyone can own one. At the moment, the primary directive of YET is on developing skills in students of the Social Sciences. Skills such as:

-   calibrating eye trackers
-   evaluating eye trackers
-   building eye trackers

    -   programming
    -   computer vision
    -   neural network
    -   Arduino
    -   3D modelling

-   experimenting with eye trackers

    -   human vision
    -   user-friendly design
    -   communication
    -   marketing

-   innovation:

    -   novel user interfaces
    -   assistive technology

## Yet

The hardware of most (not all) eye trackers is essentially a camera pointed towards the eyeball. Our first premise is that *every camera pointed to the eyeball can be an eye tracker.* Today, that creates a myriad of possibilities. 

A device that points a camera to the eye, for example by means of a head mount, is called a *Yet*. The first Yets are based on household materials and a few 3D printed parts, but more advanced head mounts are possible. The first and most recent is Yet0, which emphasize the values of simplicity and affordability. Yet0 uses a cheap USB endoscope camera, which you can buy for under 10 EUR "on the internet". Connected to a computer (or smartphone), this camera can be accessed just like a regular webcam, which makes it easy to work with in a number of ways.

Yets are numbered and the first is *Yet0*. For making Yet0, you need:

1.  a 5.5mm USB endoscope camera
2.  a 30cm transparent plastic ruler
3.  glue
4.  a tiny 3D printed part


## Yeti

The second premise of YET is that an eye tracking device consists of before-mentioned camera, plus *a software mechanism that translates a Yet stream into eye ball coordinates.* Typically this mechanism is a software that uses Computer Vision procedures for the analysis of the Yet stream. 
A *Yet interface (Yeti)* is a software that processes the *Yet stream* into eyeball coordinates (or any other useful outcome). 
Usually, Yetis are small single-purpose programs in Python, using the libraries OpenCV for stream processing and Pygame for the user interface. But, more advanced or multi-purpose interfaces are possible, as are other programming languages.

Yetis are numbered and distributed on Github ([schmettow/YET](https://github.com/schmettow/YET/tree/main/yeti)) The first is Yeti0, which does nothing more than capturing the Yet stream and displaying it. It is not interactive and therefore is build using OpenCV, only (w/o Pygame). 
The most recent Yeti for eye tracking is Yeti14, which uses nine-point calibration points and a basic statistical learning.

The programming techniques for development of Yetis are partly introduced in the following online book: [Programming for Psychologists](https://schmettow.github.io/PfP_Book/).

## Yeta

The third premise of YET is that replacing a few expensive eye trackers with an armada of Yets will be transformative for education and possibly also for research. Making eye tracking affordable for the masses will produce a myriad of 

## Definitions

1. An *eye tracker* is a device that measures the degree of rotation of the eyeball in the skull. YET0/Yet8 is an example.
2. A *head tracker* is a device that measures the position of the head in the world.
3. A *gaze tracker* is a combination of head and eye tracking that measures the perspective in the world (literally, not figuratively).
4. A device that also produces a video stream from the perspective of the user is said to *have a world view*.
5. A *world model* describes the spatial position of *areas-of-interest (AOI)* are, relative to the user's persperctive.
One simple way is to create *rectangle-of-interests (ROI)* in the world view.


# Tools

- OBS Studio

# Yet catalogue

## Yet0 (Clip-Y)

[Yet0](https://github.com/schmettow/YET/tree/main/yet/0) is a simple device intended for developing Yetis. A small 3D printed part (Clip-Y) is glued to a plastic ruler. The user has to hold the device with one hand, while using it.

The strength of Yet0 are its very low price. Clip-Y is a very simple shape that

-   used minimal material (1- 2g)
-   is almost indestructible by mechanical forces
-   can be printed without problems on *any* 3D pinter
-   requires no post-processing


# Yeti catalogue

## Yeti3

[Yeti3](https://github.com/schmettow/YET/tree/main/yeti/3) collects split-frame brightness (SFB) at random horizontal eye positions.
It produces a data set with x_pos|SBG_left|SBG_right, which can be used as training data for a model that predicts 
horizontal position by SBG. An R Markdown script is provided that loads the data set and helps identifying the most 
accurate prediction model.

## Yeti11

[Yeti11](https://github.com/schmettow/YET/tree/main/yeti/11) is the first complete interface for horizontal and vertical tracking. It uses only four calibration point and some basic arithmetics. The results are usually robust as long as movements are orthogonal, but poor on the diagonals.

## Yeti14

[Yeti14](https://github.com/schmettow/YET/tree/main/yeti/14) is the first broadly usable eye tracking engine. It implements the ideas shown in Yeti3, where split-frame brightness measures are used to train a linear model for prediction. Yeti14 splits eye frames of nine calibration points in four quadrants. With the Scikit-Learn library a basic machine learning workflow is implemented.



### Roadmaps

#### Tracking

1. Does SBG also work for vertical eye position? For a quick test, you have to flip everything around 90 degree.
1. If vertical SBG works, Yeti3 can be extended to collect data in a plane. Model selection works the same way, but needs to be extended.
Note that you get four brightness predictors.
1. Finally, the prediction model can be implemented in Python to create a full calibration procedure.

## Yeti4

In the future Yeti4 will be an upgraded version of Yeti3, that estimates the linear model by itself.

## Yeti5

CV2/Pygame Streamplayer with snaphot function

## Yeti6

Blink detection, duration measures and a ring

- eye detection
- blink detection
- timestamp method
- the eyetracker as an input device


## Yeti7

In teh future, Yeti7 is a PG/CV program to measure the accuracy of SBG

## Yeti8

[Yeti8](https://github.com/schmettow/YET/tree/main/yeti/8) performs a two-point calibration procedure.

This Yeti demonstration of horizontal brightness gradient tracking. With [Yeti3](https://github.com/schmettow/YET/tree/main/yeti/3) it could be observed that eyeball position P and brightness B are linearly related. 

    P = beta_0 * beta_1 * B

In theory that means, you only have to measure brightness at two points (a,b) and estimate the linear coefficients using the following formula:

    beta_0 = B_a
    beta_1 = (B_b - B_a) / (P_b - P_a)

Then, Yeti8 continues measuring brightness and shows a circle that follows horizontal eye movements.


### Roadmaps

#### Improve UI

1. There are many minor ways to improve the visual layout and color scheme.
1. Screen dimensions are fixed. Make them fluid (i.e. when screen dimensions change, the layout stays consistent)
1. Add a life visualization of the brighness difference


#### Improve tracking

1. The obvious improvement is to add *vertical tracking*.
1. The current algorithm uses the difference between brightness, whereas the linear model in [Yeti3](https://github.com/schmettow/YET/tree/main/yeti/3) uses L and R as separate predictors, which is the most accurate. In Python, you can implement a linear model using [Scikit.Learn](https://stackabuse.com/linear-regression-in-python-with-scikit-learn/).

#### Validate

1. Parts from Yeti3 can be used to measure accuracy.

#### Experimentation

Yeti8 is a good precursor for programming experiments, where a quick re-calibration is possible at any times.

1. Implement data collection
1. Present pictures (e.g. robot faces) and collect data
1. Present a video clip and collect data
1. Recombine Yeti8 with the Stroop experiment
1. Recombine Yeti8 with the Corsi block tapping task
1. Implement an approach-avoidance task


## Yeti9

[Yeti9](https://github.com/schmettow/YET/tree/main/yeti/9) captures the Yet stream and the cam stream (web cam).
The display can toggle face/eye detection and pair vs picture in picture layout.

### Roadmaps

#### Improve UI


#### Improve tracking

1. Yeti9 shows how face tracking works, and that can be a first step towards head tracking. 
The face usually moves when we turn our heads, and gets smaller. 
The procedure of Yeti3 can be used to measure face position/size at different 
positions. Is it linear? Then headtracking can be calibrated quickly (Yeti8)
1. OpenCV has a number of detection and tracking mechanisms. HAAR cascades are actually very advanced. 
If your YET has a discernible color, e.g. orange, then it possible to detect and track it. 
YET sits further further away from your neck than your face. It travels more and could be used as another 
predictor for head position.

#### Validate

#### Experimentation


#### Design

1. An advantage of using linear regression is that the predicted x positions come with information on uncertainty (e.g. confidence limits). This can be used to adjust the circle size in Follow mode.

## Yeti10 

Yeti10 re-iterates on the Cam stream, which strictly makes it *not a Yet interface* (NYETI).
Like in Yeti9, faces are detected. When the user takes a snapshot, the facial emotional expression 
is analyzed (library FER).



