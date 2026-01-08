Imagine
=======
A text to image SD 1.5 and SDXL GUI for machines without CUDA, AVX2, and so on. With the right model it can even work on machines without dedicated VRAM and only 8GB RAM. There are default models that should work fine, though you might not want to actually use the default LoRAs. Right now only the LCM and TCD schedulers are supported.
```
sudo apt-get install -y gir1.2-gtk-3.0 libcairo2-dev libgirepository1.0-dev libgtk-3-0 pkg-config python3-dev python3-gi python3-gi-cairo ; sudo make install ; imagine
```
