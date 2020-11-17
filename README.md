# EmotionCV

Colab between

- Wim Christiaansen [#](https://github.com/WimChristiaansen)
- Hedia Bougi [#](https://github.com/HediaBougi)
- Tomas Verrelt [#](http://github.com/tomasverrelst)
- Maarten Fish [#](https://github.com/FishMaarten)

Challenge: Train a model to pick up on sequence of emotions through video feed

# Researched topics
*Document your research here

[fastai](https://docs.fast.ai/#Learning-fastai)
# Kaggle dataset
We used [this kaggle](https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset) dataset containing ~36k images.  
The labels covered the emotions: angry, disgust, fear, happy, neutral, sad and surprise.

An accuracy of 98.3% was reached on the training set with an LSTM architecture.  
Validation on all models is rather low due to a badly labeled validation set.

# OpenCV
CV2 can be set up with a small latency which gives our model a brief window to batch the frames and process the input for a prediction.

# Contents
- [emotion_classifier.ipynb](https://github.com/FishMaarten/EmotionCV/blob/main/notebooks/emotion_classifier.ipynb) First look at the kaggle dataset
- [torch_complete.ipynb](https://github.com/FishMaarten/EmotionCV/blob/main/notebooks/torch_complete.ipynb) Complete into to PyTorch
- [emotion_classifier2.ipynb](https://github.com/FishMaarten/EmotionCV/blob/main/notebooks/emotion_classifier2.ipynb) Implementing CNN, GRU and LSTM
- [camera_stream.ipynb](https://github.com/FishMaarten/EmotionCV/blob/main/notebooks/camera_stream.ipynb) OpenCV camera feed to model predict
