# Speech Emotion Recognition Considering Nonverbal Vocalization in Affective Conversations

!["our proposed system frameworks"](https://github.com/Xuplussss/ResnetLstmSER/tree/main/non_verbal_SER/SF.png?raw=true)

## Requirements
- Python >= 2.7
- Torch >= 0.4.0

## Audio feature extraction models
prepare your data and labels in the fold "/data"
### audio emotion feature extraction model training
```
python emoRESNET_training.py
```

### sound type feature extraction model training
```
python sndRESNET_training.py
```

## emotion recognition model training
```
python ResNet_LSTM_training.py
```

## Reference
This package provides training code for the audio-visual emotion recognition paper. If you use this codebase in your experiments please cite: 

`Hsu, J. H., Su, M. H., Wu, C. H., & Chen, Y. H. (2021). Speech emotion recognition considering nonverbal vocalization in affective conversations. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29, 1675-1686.`
