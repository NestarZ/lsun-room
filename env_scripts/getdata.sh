mkdir -p data \
&& cd data \
&& wget http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/image.zip \
&& wget http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/training.mat \
&& wget http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/validation.mat \
&& wget http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/testing.mat \
&& unzip image.zip \
&& rm -rf image.zip \
&& wget http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/layout.zip \
&& unzip layout.zip \
&& rm -rf layout.zip

mkdir -p lsun_toolkit \
&& cd lsun_toolkit \
&& wget http://lsun.cs.princeton.edu/challenge/2016/roomlayout/toolkit.zip \
&& unzip toolkit.zip \
&& rm -rf toolkit.zip
