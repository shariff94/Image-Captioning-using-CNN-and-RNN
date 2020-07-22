# Image Captioning Project

In this project, I design and train a CNN-RNN (Convolutional Neural Network - Recurrent Neural Network) model for  automatically generating image captions. The network is trained on the Microsoft Common Objects in COntext [(MS COCO)](http://cocodataset.org/#home) dataset. The image captioning model is displayed below.
![Image Captioning Model](images/cnn_rnn_model.png?raw=true)

---

## Algorithm Visualization

![Encoder](images/encoder.png?raw=true)
<div align="center">The first part of the architecture i.e Encoder. A CNN structure</div>

![Decoder](images/decoder.png?raw=true)
<div align="center">The second part of the architecture i.e Decoder. A RNN structure</div>

![Encoder-Decoder](images/encoder-decoder.png?raw=true)
<div align="center">Complete architecture of CNN-RNN in tandem</div>

---

## Generating Image Captions

Here are some predictions from my model.

### Good results
![sample_172](samples/sample_172.png?raw=true)<br/>
![sample_440](samples/sample_440.png?raw=true)<br/>
![sample_457](samples/sample_457.png?raw=true)<br/>
![sample_002](samples/sample_002.png?raw=true)<br/>
![sample_029](samples/sample_029.png?raw=true)<br/>
![sample_107](samples/sample_107.png?raw=true)<br/>
![sample_202](samples/sample_202.png?raw=true)

---

## File Descriptions
- **0_Datasets.ipynb:** The purpose of this file is to initialize the COCO API and visualize the dataset. [The Microsoft Common Objects in COntext (MS COCO) dataset](https://cocodataset.org/#home) can be accessed using the COCO API. The API has methods like "getAnnIds", "loadImgs" etc to access the images and annotations. In the 0_Datasets.ipynb file we load the instance annotations and captions annotations into memory using COCO API. Then we plot a random image from the dataset, along with its five corresponding captions. This file helps in understanding the working of the COCO API and the structure of the dataset.

- **1_Preliminaries.ipynb:** The purpose of this file is to load and pre-process data from the COCO dataset and also design a CNN-RNN model for automatically generating image captions. We use the [Data loader](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) provided by pytorch to load the COCO dataset in batches. We initialize  the data loader by using the "get_loader" method in data_loader.py. The "get_loader" function takes as input a number of arguments like "transform", "mode", "batch_size" etc. Then we import the RNN decoder from model.py. It outputs a PyTorch tensor with size [batch_size, captions.shape[1], vocab_size]. The output is designed such that outputs[i,j,k] contains the model's predicted score, indicating how likely the j-th token in the i-th caption in the batch is the k-th token in the vocabulary.

- **2_Training.ipynb:** In this file, we train the encoder-decoder neural network for Image Generation.For this project, as aforementioned, the encoder is a CNN model whereas the decoder is a RNN model. The next few lines give you a brief introduction to whys and hows of the model.
    - **Encoder:**
      The CNN model we are using is the ResNet-152 network, This model is taken as it is with the only change being in the last fully connected layer. A batch normalization layer is added. The images undergo data augmentation before they are finally changed from 256 size to 224 in order to be fed into the model.

   - **Decoder:**
      It is a LSTM model(a type of LSTM model) which produces a caption by generating one word at every timestep conditioned on a context vector, the previous hidden state and the previously generated words. This model is trained from scratch.

   The optimizer used is Adam optimizer. We conclude with the training notebook here and go to the next phase.

- **3_Inference.ipynb:** The purpose of this file is to make the predictions by loading `trained model` and `vocabulary file` to get the desired result. This model generates good captions for the provided image but it can always be improved later by including hyper-parameters and using more accurate algorithms.  
  - ![sample_440](samples/sample_440.png?raw=true)<br/>
  
---

## Citation : Udacity Computer Vision Nanodegree Program
