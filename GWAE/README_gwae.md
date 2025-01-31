
Recommended to run this code under the `venv` envirionment of Python 3.10.
The requirements can be easily installed using `pip`.
```
$ python3.10 -m venv .env
$ source .env/bin/activate
(.env) $ pip install -U pip
(.env) $ pip install wheel
(.env) $ pip install -r requirements.txt
```
In `requirements.txt`, a third-party representation learning package is specified, which is downloaded from `github.com` and installed via `pip`.

## How to Train
Run `train.py` with a setting file to train models.
```
(.env) $ python train.py setting/gwae.yaml
(.env) $ python train.py setting/vae.yaml
(.env) $ python train.py setting/geco.yaml
(.env) $ python train.py setting/ali.yaml
...
```
The results are saved in the `logger_path` directory specified in the setting YAML file.

## How to Evaluate GWAEs
Run `vis.py` with the `logger_path` directory specified in the settings.
```
(.env) $ python vis.py runs/celeba_gwae
```

## How to Evaluate Models with FID
To compute the generation FID, run `fid.py` with the `logger_path` directory path.
```
(.env) $ python fid.py runs/celeba_gwae
```
*Note*: The script `fid.py` calculates the FID score of the images sampled from *the generative model*. On the other hand, the FID values computed by [the `vaetc` package](https://github.com/ganmodokix/vaetc) is *reconstruction* FID, using the test set images for generating *reconstructed* images.
