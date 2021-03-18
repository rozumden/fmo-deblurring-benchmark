Fast Moving Object (FMO) Deblurring Benchmark
==========
Simple Python library to evaluate your FMO deblurring methods.

### Usage

Implement a function that takes as input image `I [w, h, 3]`, background `B [w, h, 3]`, bounding box of approximation FMO location, the required number of generated sub-frames `n` (temporal super-resolution), and an approximate object size. Your method should output temporal super-resolution mini-video of size `[w, h, 3, n]`. Optionally, if you want to evaluate trajectory accuracy, output the sub-frame object trajectory of size `[2, n]` or `None`.

An example of a dummy algorithm that always outputs the input image:

```python
def my_deblur(I,B,bbox,nsplits,radius):
        return np.repeat(I[:,:,:,None], nsplits, 3), None
```

### Baselines

We provide several baseline and state-of-the-art methods.

## Dummy baselines

Two baselines, one that always outputs the input image, and another that output the background image. Example is shown in `example_dummy.py`.

