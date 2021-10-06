---
title: "Your Eye Tracker developers guide"
author: "M Schmettow"
date: "30/09/2021"
output: html_document
---

# The Your-eye-tracker (YET) project

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

Most (not all) eye trackers are devices that use a camera to record the eye ball. The first idea of YET is that *every camera can be an eye tracker.* The first designs in YET use a USB endoscope camera with 5.5mm or 7mm diameter, which you can buy for under 10 EUR "on the internet". A combination of a camera module and a head mount is called a *Yet*. The first Yets will be based on household materials and a few 3D printed parts, but more advanced head mounts are possible.

Yets are numbered and the first is *Yet0*. For making Yet0, you need:

1.  a 5.5mm USB endoscope camera
2.  a 30cm transparent plastic ruler
3.  glue
4.  a tiny 3D printed part, called *Clip-Y*

## Yeti

The second idea of YET is that *an eye tracking device consists of before-mentioned camera, plus a software mechanism that translates a Yet stream into eye ball coordinates.* Typically this mechanism is a software that uses Computer Vision procedures for the analysis of the Yet stream. A *Yet interface (Yeti)* is a software that processes the *Yet stream* of frames in a useful or interesting way. Usually, Yetis are self-written single-purpose programs in Python, using the libraries OpenCV for stream processing and Pygame for the user interface. But, more advanced or multi-purpose interfaces are possible.

Yetis are numbered and distributed on Github ([schmettow/YET](https://github.com/schmettow/YET/tree/main/yeti)) The first is *Yeti0, which* does nothing more than capturing the Yet stream and displaying it. It is not interactive and therefore is build using OpenCV (w/o Pygame). The first interactive Yeti to use Pygame is Yeti2, which introduces a very simple algorithm (split-frame brightness gradient, SBG), as well as a common structure for Yetis.

The programming techniques for development of Yetis will be introduced in the following online book: [Programming for Psychologists](https://schmettow.github.io/PfP_Book/).

## Yeta

The third idea of YET is that replacing a few expensive eye trackers with *an armada of cheap ones, can be transformative for research*. In the future, we expect to be able to build *Yet appliances (Yeta)*, such as mobile recording devices, novel user interfaces or assistive technologies.

# Resources

Public Github project [schmettow/YET](https://github.com/schmettow/YET/)

# Yet catalogue

## Yet0 (Clip-Y)

[Yet0](https://github.com/schmettow/YET/tree/main/yet/0) is a simple device intended for developing Yetis. A small 3D printed part (Clip-Y) is glued to a plastic ruler. The user has to hold the device with one hand, while using it.

The strength of Yet0 are its very low price. Clip-Y is a very simple shape that

-   used minimal material (1- 2g)

-   is almost indestructible by mechanical forces

-   can be printed without problems on *any* 3D printer

-   requires no post-processing

# Yeti catalogue

## Yeti2

[Yeti2](https://github.com/schmettow/YET/tree/main/yeti/2) is a demonstration of brightness gradient estimation for eye tracking.

-   uses OC and PG
-   split-frame brightness gradient
-   needs calibration by setting variables
-   connects with Yeti3 for automated calibration


## Yeti8

[Yeti8](https://github.com/schmettow/YET/tree/main/yeti/8) performs a two-point calibration procedure.

This Yeti demonstration of horizontal brightness gradient tracking. With [Yeti3](https://github.com/schmettow/YET/tree/main/yeti/3) it could be observed that eyeball position x and brightness y are linearly related. 

    x = beta_0 * beta_1 * y

In theory that means, you only have to measure brightness at two points (A,B) and estimate the linear coefficients using the following formula:

    beta_0 = y_A
    beta_1 = (y_B - y_A) / (x_B - x_A)

Then, Yeti8 continues measuring brightness and shows a circle that follows horizontal eye movements.


### Roadmaps


#### Improve tracking

1. The obvious improvement is to add *vertical tracking*.
1. The current algorithm uses the difference between brightness, whereas the linear model in [Yeti3](https://github.com/schmettow/YET/tree/main/yeti/3) uses L and R as separate predictors, which is the most accurate. In Python, you can implement a linear model using, for example,  [Scikit.Learn](https://stackabuse.com/linear-regression-in-python-with-scikit-learn/).

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

#### Assistive technology



#### Design

1. An advantage of using linear regression is that the predicted x positions come with information on uncertainty (e.g. confidence limits). This can be used to adjust the circle size in Follow mode.

    


