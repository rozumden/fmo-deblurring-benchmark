Fast Moving Object (FMO) Deblurring Benchmark
==========
Simple Python library to evaluate your FMO deblurring methods.

### Datasets

All three datasets (TbD, TbD-3D, Falling Objects) can be downloaded by running (after modifying the data storage folder path):
```bash
bash download_datasets.sh
```

### Usage

Implement a function that takes as input image `I [w, h, 3]`, background `B [w, h, 3]`, bounding box of approximation FMO location, the required number of generated sub-frames `n` (temporal super-resolution), and an approximate object size. Your method should output temporal super-resolution mini-video of size `[w, h, 3, n]`. Optionally, if you want to evaluate trajectory accuracy, output the sub-frame object trajectory of size `[2, n]` or `None`.

An example of a dummy algorithm that always outputs the input image and does not evaluate the trajectory accuracy:

```python
def my_deblur(I,B,bbox,nsplits,radius):
        return np.repeat(I[:,:,:,None], nsplits, 3), None
```

Baselines
------------

We provide several baseline and state-of-the-art methods.

#### Dummy baselines

Two baselines, one that always outputs the input image, and another that output the background image. Example is shown in `example_dummy.py`.


#### Deblatting

To evaluate this method, please check out the [deblatting sub-module](https://github.com/rozumden/deblatting_python). We provide three versions of deblatting: classical deblatting with single appearance (TbD), deblatting with chanring appearance (TbD-3D), and deblatting with trajectory oracle (TbD-O). Examples are shown in `example_deblatting.py`.

#### DeFMO - current state-of-the-art

To evaluate this method, please check out the [DeFMO sub-module](https://github.com/rozumden/DeFMO). Example is shown in `example_defmo.py`.

Scores
------------
TbD-3D-Oracle has access to the ground-truth trajectory. Therefore, it's not a competitive baseline and is provided just for the reference.
#### Falling Objects dataset
Arbitrary shaped and textured objects.

| Score | Bg         |    Im | Jin et al. | DeblurGAN-v2 | TbD    | TbD-3D | DeFMO      | (TbD-3D-Oracle) 
| ----- | :---:      | :---: | :---:      | :---:        | :-:    | :---:  | :---:      | :---: 
| TIoU  |  0         | 0     |        0   |      0       | 0.539  | 0.539  | **0.684**  | 1.000
| PSNR  | 19.71      | 23.76 |  23.54     |    23.36     | 20.53  | 23.42  | **26.83**  | 22.82
| SSIM  | 0.456      | 0.594 | 0.575      |   0.588      | 0.591  | 0.671  | **0.753**  | 0.701  

#### TbD-3D dataset
Mostly spherical but significantly textured objects.

| Score | Bg         |    Im | Jin et al. | DeblurGAN-v2 | TbD    | TbD-3D | DeFMO      | (TbD-3D-Oracle) 
| ----- | :---:      | :---: | :---:      | :---:        | :-:    | :---:  | :---:      | :---: 
| TIoU  |  0         | 0     |        0   |      0       | 0.598  | 0.598  | **0.879**  | 1.000
| PSNR  | 19.81      | 24.80 | 24.52      | 23.58        | 18.84  | 23.13  | **26.23**  | 24.63
| SSIM  |  0.426     | 0.640 | 0.590      | 0.603        | 0.504  | 0.651  | **0.699**  | 0.703


#### TbD dataset
Mostly spherical and uniformly colored objects.

| Score | Bg         |    Im | Jin et al. | DeblurGAN-v2 | TbD    | TbD-3D   | DeFMO      | (TbD-3D-Oracle) 
| ----- | :---:      | :---: | :---:      | :---:        | :-:    | :---:    | :---:      | :---: 
| TIoU  |  0         | 0     |        0   |      0       | 0.542  | 0.542    | **0.550**  | 1.000
| PSNR  | 21.48      | 25.06 | 24.90      | 24.27        | 23.22  | 25.21    | **25.57**  |  26.23
| SSIM  |  0.466     | 0.568 | 0.530      | 0.537        | 0.605  |**0.674** |  0.602     | 0.712

Reference
------------
If you use this repository, please cite the following [publication](https://arxiv.org/abs/2012.00595):

```bibtex
@inproceedings{defmo,
  author = {Denys Rozumnyi and Martin R. Oswald and Vittorio Ferrari and Jiri Matas and Marc Pollefeys},
  title = {DeFMO: Deblurring and Shape Recovery of Fast Moving Objects},
  booktitle = {CVPR},
  address = {Nashville, Tennessee, USA},
  month = jun,
  year = {2021}
}
```
The baseline TbD method:
```bibtex
@inproceedings{Kotera-et-al-ICCVW-2019,
  author = {Jan Kotera and Denys Rozumnyi and Filip Sroubek and Jiri Matas},
  title = {Intra-frame Object Tracking by Deblatting},
  booktitle = {Internatioal Conference on Computer Vision Workshop (ICCVW), 
  Visual Object Tracking Challenge Workshop, 2019},
  address = {Seoul, South Korea},
  month = oct,
  year = {2019}
}
```
The baseline TbD-3D or TbD-O method:
```bibtex
@inproceedings{Rozumnyi-et-al-CVPR-2020,
  author = {Denys Rozumnyi and Jan Kotera and Filip Sroubek and Jiri Matas},
  title = {Sub-frame Appearance and 6D Pose Estimation of Fast Moving Objects},
  booktitle = {CVPR},
  address = {Seattle, Washington, USA},
  month = jun,
  year = {2020}
}
```
Some ideas are taken from:
```bibtex
@inproceedings{Rozumnyi-et-al-CVPR-2017,
  author = {Denys Rozumnyi and Jan Kotera and Filip Sroubek and Lukas Novotny and Jiri Matas},
  title = {The World of Fast Moving Objects},
  booktitle = {CVPR},
  address = {Honolulu, Hawaii, USA},
  month = jul,
  year = {2017}
}
