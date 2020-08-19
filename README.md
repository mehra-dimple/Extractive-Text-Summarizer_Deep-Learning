# Extractive Text Summarization

## Project Title ##  
Extractive Text Summarization  
(An approach for generating short and precise summaries for long text documents using the implementation of Deep Belief Neural Networks)

## Project Idea Description ## 
The size of information on the internet is increasing. It has become tough for the users to dig into the loads of information to analyze it and draw conclusions. Earlier, humans used to summarize the text by their own, but today due to increasing data, it is difficult for the human beings to cope up with the huge data due to which the time required for the users to summarize and analyse the huge data is increased.

Convolutional neural networks (CNN) is originally proposed for computer vision by LeCun et al. It produces excellent results for computer vision tasks and recently has been proven to be effective for various NLP tasks as well, such as POS tagging, sentence modeling, semantic embedding and sentence classification, to name a few. 

Text summarization solves this problem by generating a summary, selecting sentences which are most important from the document without losing the information. The solution to reduce reading time of the user is producing a succinct document summary.

## Abstract ##  
Text Summarization is the process of obtaining salient information from an authentic text document. In this technique, the extracted information is achieved as a summarized report and conferred as a concise summary to the user; it is challenging for the user to verve through altogether the information accessible on web.

Reduction of text is a very complex problem which, in spite of the progress in the area thus far, poses many challenges to the scientific community. It is also relevant application in today’s information society given the exponential growth of textual information online and the need to promptly assess the contents of text collections. It has long been assumed that summarization presupposes to understand the input text, which means for identifying the important point of the document, explicit representation of the text must be calculated therefore, text summarization became an interesting application to test the understanding capabilities of artificial systems.


## Approach ##  
The objective of this project building a neural network with TensorFlow that can create relevant summaries for the dataset. To build our model we will use a two-layered bidirectional RNN with LSTMs on the input data and two layers, each with an LSTM on the target data. The deep network power is capable of
1. analysing sequences of input
2. understanding text
3. outputting sequences of output in form of summarizes
hence the name of sequence 2 sequence(seq2seq), sequence of inputs to sequence of outputs, which is the main algorithm that we use in building the model.



## Architecture Diagram ##

![](FrontEnd/Architecture.jpg)


## Persona ##

![](FrontEnd/Police.png)


![](FrontEnd/EMT.png)


## Technology Stack ##
* 	Deep CNN to classify the sound.
* 	Log scaled mel-spectrogram to observe time frequency patches.
* 	Machine-Learning : Python 
* 	Keras for Deep Learning Model implementation
* 	Librosa for analysis of audio signals.
*   IBM cloud to host the front-end
*   AWS cloud to host the middleware and back-end
*   Angular JS and AJAX based User-Interface

## Video Trailer ##

https://app.vyond.com/videos/a754695e-ceca-4c6b-9224-344f87f3fcf0

## Link to HomePage ##

http://79.5c.c1ad.ip4.static.sl-reverse.com.:31963/

## Home Page ##

![](FrontEnd/sp.png)

## Email Alert for Gunshot ##

![](Alert%20Messages/Email%20alert.jpeg)

## Text Message Alert for Gunshot ##

![](Alert%20Messages/Message%20Alert.jpeg)
