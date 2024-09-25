# DomainForensics: Exposing Face Forgery across Domains via Bi-directional Adaptation

> The official PyTorch implementation for the following paper:
>
> [DomainForensics: Exposing Face Forgery Across Domains via Bi-Directional Adaptation](https://ieeexplore.ieee.org/abstract/document/10601589)
> 
> Qingxuan Lv; Yuezun Li*; Junyu Dong*; Sheng Chen; Hui Yu; Huiyu Zhou; Shu Zhang
> 
> IEEE Transactions on Information Forensics and Security

# Recomended Development Environment
## Basic dependencies
```
opencv-python
numpy
easydict
pytorch-lightning==1.7.5
rich
matplotlib
```
## Other dependencies
### RetinaFace-pytorch
install [retinaface-pytorch](https://github.com/biubug6/Pytorch_Retinaface).
### Jpeg-turbo
install [Libjpeg-Turbo](https://github.com/libjpeg-turbo/libjpeg-turbo).
### Jpeg2dct
install [jpeg2dct](https://github.com/uber-research/jpeg2dct)



# Setup
## Datasets
### FF++
1. Download FF++ from [ff++ github](https://github.com/ondyari/FaceForensics)
2. Extract to `/path/to/FF++`
3. Crop faces by using `utils/crop_retinaface_ff.py` e.g. `python crop_retinaface_ff.py -d Deepfakes -n 8`

### Celeb-DF
1. Download Celeb-DF v2 from [celeb-df github](https://github.com/yuezunli/celeb-deepfakeforensics)
2. Extract to `/path/to/Celeb-DF`
3. Crop faces by using `utils/crop_retinaface_ff.py` e.g. `python crop_retinaface_ff.py -d Celeb-real -n 8 -s train`


## Training
> We train the model on two RTX2080Ti with 11Gx2 GPU memory.

Change the config within `train.py` for different adaptation setting.
e.g. 
    
``` 
    # from Deepfakes to Face2Face
    cfg.DATAS.SOURCE = ["Deepfakes"]
    cfg.DATAS.TARGET = ['Face2Face']
```

run training by executing `python train.py`

## Testing
1. Change the `cfg.TESTING.MODEL_WEIGHT` to the pretrained weight
2. run testing by executing `python testing.py`

## Pretrained Models

|Task|Dataset|AUC|Model|Size|
|:-:|:-:|:-:|:-:|:-:|
|FaceSwap -> Face2Face| FF++ | 99.13 | [google drive](https://drive.google.com/file/d/1XwXDSq5XlH1XlBdmbZnGKIttbnAGsO5l/view?usp=sharing)| 855MB |
| NeuralTextures -> FaceSwap| FF++ | 97.67 | [google drive](https://drive.google.com/file/d/1sUG6vXBfHTMzIKNu_9ewxGINgdKP-SDv/view?usp=sharing)| 855MB |



# Citation
```
@ARTICLE{10601589,
  author={Lv, Qingxuan and Li, Yuezun and Dong, Junyu and Chen, Sheng and Yu, Hui and Zhou, Huiyu and Zhang, Shu},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={DomainForensics: Exposing Face Forgery Across Domains via Bi-Directional Adaptation}, 
  year={2024},
  volume={19},
  number={},
  pages={7275-7289},
  keywords={Forgery;Deepfakes;Detectors;Feature extraction;Faces;Training;Bidirectional control;Digital forensics;DeepFake detection;DomainForensics},
  doi={10.1109/TIFS.2024.3426317}}
```
