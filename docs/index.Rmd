---
title: "Your Eye Tracker developers guide"
author: "M Schmettow"
date: "30/09/2021"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# What are YET, Yets, Yetis and Yetas?

YET is Your Eye Tracker. The goal of this project is to make eye trackers so cheap that everyone can own one. Currently, YET focusses on developing skills in students of the Social Sciences. Skills such as:

+ building research instruments
+ programming research instruments
+ running experiments with eye trackers
+ eye tracker for introspection
+ calibrating an eye tracker
+ evaluating an eye tracker
+ improving an eye tracker


One idea of YET is that the hardware of an eye tracker is nothing more but a camera pointed to an eye. YET uses a USB endoscope with 5.5mm or 7mm diameter, which you can buy for 5 - 7 EUR "on the internet". A *Yet* is a head mount for such a camera. A Yet consists of 3D printed parts and household materials, such that it can be produced easily.

The second idea of YET is that an eye tracking *device* consists of before-mentioned camera, plus a mechanism that translates a Yet stream into eye ball coordinates. Typically this nechanism is a software that uses Computer Vision procedures for the analysis of the Yet stream. A *Yet interface (Yeti)* is a  software that processes the Yet stream in a useful or interesting way. Usually, Yetis are small self-written programs in Python, using the libraries OpenCV for stream processing and Pygame for the user interface.

The third idea of YET is that having an armada of cheap eye trackers, rather than a few expensive ones, can be transformative for research with eye trackers. In the future, we expect to be able to build *Yet appliances (Yeta)*, where Yets are connected to mobile recorders, integrated with other sensor, or employ advanced  machine learning techniques, for example to automatically build a world model.

