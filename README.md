"duplicates.zip" is a folder that contains five unique cat images, four unique dog images, and overall 16 images (including seven duplicates images). These duplicates images are generated by varying dimensions, intensity, and color of existing unique photos. <br> 

```ruby
colab$ !unzip "/content/duplicates.zip" 
```
```ruby
colab$ ls -l duplicates/*.jpg duplicates/*.png | wc -l  #count total image files with .jpg and .png
```

Execute the duplicate_images_remover.py with required command line arguments folder_path, hamming distance as threshold and details printing as True/False <br>
```ruby
colab$ %run duplicate_images_remover.py -f /content/duplicates -t 2 -i false
```

<img width="280" alt="duplicate_img_remover" src="https://user-images.githubusercontent.com/18000553/134928552-c9acde2b-2eef-42ac-bda1-f62565300862.png">

One may also explore the Structural Similarity Index (https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html#sphx-glr-auto-examples-transform-plot-ssim-py)

# Efficient Remover of Duplicate Images
