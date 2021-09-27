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

<img width="266" alt="duplicate_img_remover" src="https://user-images.githubusercontent.com/18000553/134904064-9ed456aa-ca32-4efe-9a8e-139fb4f59d48.png">

# Efficient Remover of Duplicate Images
