# Hyperspectral Image Classification Using Random Occlusion Data Augmentation
The Code for "Hyperspectral Image Classification Using Random Occlusion Data Augmentation". [https://ieeexplore.ieee.org/document/8694852]
```
J. M. Haut, M. E. Paoletti, J. Plaza, A. Plaza and J. Li.
Hyperspectral Image Classification Using Random Occlusion Data Augmentation.
IEEE Geoscience and Remote Sensing Letters.
DOI: 10.1109/LGRS.2019.2909495 
vol. 16, no. 11, pp. 1751-1755, November 2019.
```

![ROhsi](https://github.com/mhaut/ROhsi/blob/master/images/rohsi.png)



## Example of use
```
# Without datasets
git clone https://github.com/mhaut/ROhsi

# With datasets
git clone --recursive https://github.com/mhaut/ROhsi
cd HSI-datasets
python join_dsets.py
```

### Run code

```
Indian Pines
python main.py --tr_percent 0.15 --dataset IP --verbose # without data augmentation
python main.py --p 0.25 --tr_percent 0.15 --dataset IP --verbose # with data augmentation
python main.py --p 0.5 --tr_percent 0.15 --dataset IP --verbose # with data augmentation

University of Pavia
python main.py --tr_percent 0.10 --dataset PU --verbose # without data augmentation
python main.py --p 0.25 --tr_percent 0.10 --dataset PU --verbose # with data augmentation
python main.py --p 0.5 --tr_percent 0.10 --dataset PU --verbose # with data augmentation
```
