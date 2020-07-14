# kinetics-download

Simple tool to download videos using proxy from [kinetics dataset](https://deepmind.com/research/open-source/open-source-datasets/kinetics/).

Also have functionality to trim downloaded videos to the length of action. To support trimming, ffmpeg should be installed and added to environment variable PATH.

## Usage
Install requirements first
```
pip install -r requirements.txt
```
Install [youtube-dl](https://youtube-dl.org/) and [ffmpeg](https://github.com/FFmpeg/FFmpeg) according to your system.

You should download and unzip csv files with links to videos. You can download such files [here](https://deepmind.com/research/open-source/open-source-datasets/kinetics/).
For example here is link to [kinetics-600 training.zip](https://deepmind.com/documents/193/kinetics_600_train%20(1).zip)

### Downloading
failure_list.txt stores unavaliable links.
```
cd kinetics-download
python download.py failure_list.txt /path/to/kinetic_train.csv /path/to/videos/
```

### Downloading and trimming
```
cd kinetics-download
python download.py failure_list.txt /path/to/kinetic_train.csv /path/to/videos/ --trim
```

### Parallel processing
One can run downloading and trimming in several processes. To do so, just add flag --num-jobs with number of jobs running in parallel.
For example:
```
cd kinetics-download
python download.py failure_list.txt /path/to/kinetic_train.csv /path/to/videos/ --trim --num-jobs 10
```

### Trimming with your Nvidia device
You can execute ffmpeg trimming with cuda accelerating:
```
cd kinetics-download
python download.py failure_list.txt /path/to/kinetic_train.csv /path/to/videos/ --trim --num-jobs 10 --NV
```

### With proxy
We add the specified HTTP/HTTPS/SOCKS proxy support for users cannot visit YouTube by direct connection.
To enable SOCKS proxy, specify a proper scheme. For example socks5://127.0.0.1:1080/. Pass in an empty string (--proxy "") for direct connection').
For example:
```
cd kinetics-download
python download.py failure_list.txt /path/to/kinetic_train.csv /path/to/videos/ --trim --num-jobs 10 --proxy https://127.0.0.1:8087
```

### Reference
This project is mostly derived from
https://github.com/piaxar/kinetics-downloader
