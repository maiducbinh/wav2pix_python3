import csv
import yt_dlp
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-url_csv", required=True,
                    help="CSV file where the url of the videos to download are.")
parser.add_argument("-dataset_path", required=True,
                    help="Output folder of the dataset divided into /video and /audio.")
args = parser.parse_args()


def read_channels(file_path):
    parsed_channels = []
    with open(file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            parsed_channels.append({
                "name": row["Name"],
                "gender": row["Gender"],
                "url": row["Channel-URL"]
            })
    return parsed_channels


def download(url_list):
    error_counter = 0
    count = 0

    for url in url_list:
        count += 1
        print(f"Downloading videos and audios {count}/{len(url_list)} with url [{url['url']}] from {url['name']}")
        out_path = os.path.join(args.dataset_path, url['name'])
        video_out_path = os.path.join(out_path, 'video')
        audio_out_path = os.path.join(out_path, 'audio')
        
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(video_out_path, exist_ok=True)
        os.makedirs(audio_out_path, exist_ok=True)

        try:
            video_options = {
                'format': '135+140',  # 480p video + m4a audio
                'verbose': True,
                'continue': True,
                'ignoreerrors': True,
                'no_overwrites': True,
                'sleep_interval': 5,
                'playliststart': 1,
                'playlistend': 15,
                'extract_audio': True,
                'writeautomaticsub': True,
                'outtmpl': '%(title)s.%(ext)s',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            }

            with yt_dlp.YoutubeDL(video_options) as ydl:
                ydl.download([url['url']])

        except Exception as e:
            print(f"Download error: {str(e)}")
            error_counter += 1
            continue

        # Move downloaded files to appropriate directories
        workdir = os.listdir('./')
        for file in workdir:
            try:
                if file.endswith((".mp4", ".webm")):
                    shutil.move(file, os.path.join(video_out_path, file))
                elif file.endswith((".m4a", ".wav", ".mp3")):
                    shutil.move(file, os.path.join(audio_out_path, file))
            except Exception as e:
                print(f"Error moving file {file}: {str(e)}")

    print(f"Found {error_counter} errors")


if __name__ == "__main__":
    urls = read_channels(args.url_csv)
    download(urls)
