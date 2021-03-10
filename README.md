# HyACS

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Authors](#authors)

## General info
HyACS is a software that performs an algorithm to identify, count, and extract some physical features of Hyalella individuals by applying artificial vision and digital image processing techniques. The images used correspond to Petri dishes with these organisms inside. 

Hyalella is a bioindicator organism that has a high organic material content, and it is used in ecotoxicological tests due to its high sensitivity to heavy metals and environmental impacts. Hyalella's high population is related to fluctuations in electrical conductivity and low concentrations of dissolved oxygen density, which allows determining changes in dominance, diversity, reproductive strategies, and their relationship with physical and chemical variables of the environment.

This software takes an image of a Petri dish and processes it performing the predictions to identify and count Hyalellas, and then calculates some of the individual’s metrics such as arc length, perimeter, eccentricity, etc.
	
## Technologies
This project was created with:
#### Hardware:
* A laptop with an Intel ® CoreTM i5-8250U processor, 8 GB of RAM, 64-bit, 2TB of HDD, an Intel ® UHD Graphics 620 graphics card of 4GB, and Windows 10 as the operating system.

#### Software
* Python version: 3.7.3.
* PyQt5 version: 5.12.3.
* OpenCV module version: 4.1.0. 
* Qt Designer version: 5.9.7.
* IDE: Spyder 3.3.6 Anaconda.
	
## Setup
After having the project files downloaded or cloned in a folder, open the Anaconda prompt and go to that path using the cd command:
```
$ cd your-path-here/HyACS
```

There are several files related to the graphical user interface and the deep learning model, but 'HyACS.py' is the main file, so run it by using the Anaconda prompt typing:
```
$ python HyACS.py
```
Then, the initial interface will appear. You can follow the user's guide to continue using the software.

## Authors
* David Stephen Fernández Mc Cann
* Fabio de Jesús Vélez Macias
* Nestor Jaime Aguirre Ramírez 
* Julio Eduardo Cañón Barriga
* Ludy Yanith Pineda Alarcón 
* Yarin Tatiana Puerta Quintana 
* Maycol Esteban Zuluaga Montoya 
* Santiago Ruiz González

