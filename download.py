import pandas as pd
import argparse
import os
import shutil
import subprocess
from joblib import delayed
from joblib import Parallel
import cv2 



REQUIRED_COLUMNS = ['label', 'youtube_id', 'time_start', 'time_end', 'split', 'is_cc']
TRIM_FORMAT = '%06d'
URL_BASE = 'https://www.youtube.com/watch?v='

VIDEO_EXTENSION = '.mp4'
VIDEO_FORMAT = 'mp4'
TOTAL_VIDEOS = 0


def create_file_structure(path, folders_names):
    """
    Creates folders in specified path.
    :return: dict
        Mapping from label to absolute path folder, with videos of this label
    """
    mapping = {}
    if not os.path.exists(path):
        os.mkdir(path)
    for name in folders_names:
        dir_ = os.path.join(path, name)
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        mapping[name] = dir_
    return mapping


def download_clip(row, label_to_dir, trim, count, flist, proxy, NV):
    """
    Download clip from youtube.
    row: dict-like objects with keys: ['label', 'youtube_id', 'time_start', 'time_end']
    'time_start' and 'time_end' matter if trim is True
    trim: bool, trim video to action ot not
    """

    label = row['label']
    filename = row['youtube_id']
    time_start = row['time_start']
    time_end = row['time_end']

    # if trim, save full video to tmp folder
    output_path = label_to_dir['_tmp'] if trim else label_to_dir[label]
    start = str(time_start)
    end = str(time_end - time_start)
    output_filename = os.path.join(label_to_dir[label],
                               filename + '_{}_{}'.format(start, end) + VIDEO_EXTENSION)

    # don't download if already exists
    if os.path.exists(output_filename):
        cap = cv2.VideoCapture(output_filename)
        ret,frame = cap.read()
        if ret:
            cap.release()
            return 
        cap.release()

    if os.path.exists(os.path.join(output_path, filename + VIDEO_EXTENSION)):
        cap = cv2.VideoCapture(os.path.join(output_path, filename + VIDEO_EXTENSION))
        ret,frame = cap.read()
        cap.release()
    else:
        ret = False
    if not ret:
        try:
            commond = 'youtube-dl --no-continue ' + URL_BASE + filename + \
            ' -f "best[height<=480]" -o ' + os.path.join(output_path, filename + VIDEO_EXTENSION) \
            + ' --proxy ' + proxy
            subprocess.check_output(commond, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            with open(flist, 'a') as f:
                f.write(URL_BASE + filename+'\n')
            print('Unavailable video: ', filename)
            return
    

    if os.path.exists(os.path.join(output_path, filename + VIDEO_EXTENSION)) and trim:
        # Take video from tmp folder and put trimmed to final destination folder
        # better write full path to video
        input_filename = os.path.join(output_path, filename + VIDEO_EXTENSION)

        # Construct command to trim the videos (ffmpeg required).
        if NV:
            command_GPU = 'ffmpeg -hwaccel cuvid -y -i "{input_filename}" ' \
                  '-ss {time_start} ' \
                  '-t {time_end} ' \
                  '-c:v h264_nvenc -c:a copy -threads 1 ' \
                  '"{output_filename}"'.format(
                       input_filename=input_filename,
                       time_start=start,
                       time_end=end,
                       output_filename=output_filename
                   )
        command_CPU = 'ffmpeg -y -i "{input_filename}" ' \
                  '-ss {time_start} ' \
                  '-t {time_end} ' \
                  '-c:v libx264 -c:a copy -threads 1 ' \
                  '"{output_filename}"'.format(
                       input_filename=input_filename,
                       time_start=start,
                       time_end=end,
                       output_filename=output_filename
                   )
        if NV:
            try:
                subprocess.check_output(command_GPU, shell=True, stderr=subprocess.STDOUT)
                print('Processed %i out of %i' % (count + 1, TOTAL_VIDEOS))
            except subprocess.CalledProcessError:
                with open(flist, 'a') as f:
                    f.write(URL_BASE + filename+'\n')
                print('Error while trimming: ', filename)
                return False
        else:
            try:
                subprocess.check_output(command_CPU, shell=True, stderr=subprocess.STDOUT)
                print('Processed %i out of %i' % (count + 1, TOTAL_VIDEOS))
            except subprocess.CalledProcessError:
                with open(flist, 'a') as f:
                    f.write(URL_BASE + filename+'\n')
                print('Error while trimming: ', filename)
                return False

    


def main(input_csv, output_dir, trim, num_jobs, flist, proxy, NV, start, stop):

    global TOTAL_VIDEOS

    assert input_csv[-4:] == '.csv', 'Provided input is not a .csv file'
    links_df = pd.read_csv(input_csv)
    assert all(elem in REQUIRED_COLUMNS for elem in links_df.columns.values),\
        'Input csv doesn\'t contain required columns.'

    # Creates folders where videos will be saved later
    # Also create '_tmp' directory for temporary files
    folders_names = links_df['label'].unique().tolist() + ['_tmp']
    label_to_dir = create_file_structure(path=output_dir,
                                         folders_names=folders_names)
    with open(flist, 'w') as f:
        f.write('')

    videos_to_download = links_df[start:stop]
    TOTAL_VIDEOS = videos_to_download.shape[0]
    # Download files by links from dataframe
    Parallel(n_jobs=num_jobs)(delayed(download_clip)(
            row, label_to_dir, trim, count, flist, proxy, NV) for count, row in videos_to_download.iterrows())


if __name__ == '__main__':
    description = 'Script for downloading and trimming videos from Kinetics dataset.' \
                  'Supports Kinetics-400 as well as Kinetics-600.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('flist', default='failure_list.txt' ,type=str,
                   help=('Path to failure list file'))
    p.add_argument('input_csv', type=str,
                   help=('Path to csv file, containing links to youtube videos.\n'
                         'Should contain following columns:\n'
                         'label, youtube_id, time_start, time_end, split, is_cc'))
    p.add_argument('output_dir', type=str,
                   help='Output directory where videos will be saved.\n'
                        'It will be created if doesn\'t exist')
    p.add_argument('--trim', action='store_true', dest='trim', default=False,
                   help='If specified, trims downloaded video, using values, provided in input_csv.\n'
                        'Requires "ffmpeg" installed and added to environment PATH')
    p.add_argument('--NV', action='store_true', dest='NV', default=False,
                   help='True: Use ffmpeg nvenc')
    p.add_argument('--start', type=int, default=0,
                   help='Start download from middle of the index in csv.')
    p.add_argument('--stop', type=int, default=None,
                   help='Stop index.')
    p.add_argument('--num-jobs', type=int, default=1,
                   help='Number of parallel processes for downloading and trimming.')
    p.add_argument('--proxy', type=str, default="socks5://127.0.0.1:10808",
                   help='Use the specified HTTP/HTTPS/SOCKS proxy. \n'
                        'To enable SOCKS proxy, specify a proper scheme.\n'
                        'For example socks5://127.0.0.1:1080/. \n'
                        'Pass in an empty string (--proxy "") for direct connection.')
    main(**vars(p.parse_args()))
