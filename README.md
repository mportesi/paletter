# Paletter.py
A simple Python script that use the K-Means algorithm to generate a color palette from an image.
Using openCV, scikit-learn and numpy.

![Everyone](https://github.com/mportesi/paletter/blob/main/generated_palette/howl_castle_palette.png)

## Usage

Run it from the command line:

```bash
$ python paletter.py C:/path-to/myimage.png
```
Various arguments can be specified:

```bash
-n 9 # number of color the palette will show
--border 20 # white border size. Default 20.
--margin 2 # whitespace between color in the palette. Default 2
-p, --palette 0 # generate only the palette without the original image attached. Default 0 (1 for using it).
```

## To-do
- Finish readme
- clean up code
- implement showing the color hex values in the palette
- better console management / argparse description

