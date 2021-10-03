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

-   usesd minimal material (1- 2g)

-   is almost indestructible by mechanical forces

-   can be printed without problems on *any* 3D printer

-   requires no post-processing

# Yeti catalogues

## Yeti2

[Yeti2](https://github.com/schmettow/YET/tree/main/yeti/2) is a demonstration of brightness gradient estimation for eye tracking.

-   uses OC and PG
-   split-frame brightness gradient
-   needs calibration by setting variables
-   connects with Yeti3 for automated calibration

# 
