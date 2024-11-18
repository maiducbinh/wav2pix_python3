import cv2
import argparse
import pickle
import math
import numpy as np
import skvideo.io
import os
import wave
import csv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-dataset_path", required=True,
                    help="Parent directory where the dataset is stored")
parser.add_argument("-confidence_thresh", required=True,
                    help="Threshold of confidence to keep bounding boxes")
parser.add_argument("-haarcascade_path", required=True,
                    help="Path of the haarcascade XML file")
args = parser.parse_args()


def get_audio_samples(wav_file):
    """Function that reads a WAV file as a numpy array."""
    with wave.open(wav_file, 'rb') as wavfile:
        length = wavfile.getnframes()
        wave_data = wavfile.readframes(length)
    return np.frombuffer(wave_data, dtype=np.int16)


def cut_audio_frame(wavfile_data, current_frame_idx, fps):
    frate = 16000
    start_time = (current_frame_idx - fps * 4) / fps
    start_time = max(start_time, 0) * frate
    final_time = (current_frame_idx + fps * 4) / fps * frate
    return wavfile_data[int(start_time):int(final_time)]


def check_audio_frames(faces_frames_path, audio_frames_path):
    """Function that checks if audio cropping has been done correctly and, if not, it corrects it."""
    print('Checking audio and video crops sync...')
    faces_frames_ids = [int(x.split('_')[-1].split('.')[0]) for x in os.listdir(faces_frames_path)]
    for audio_file in os.listdir(audio_frames_path):
        audio_id = int(audio_file.split('_')[-1].split('.')[0])
        if audio_id not in faces_frames_ids:
            os.remove(os.path.join(audio_frames_path, audio_file))
    print('Done!')


def read_channels(file_path):
    parsed_channels = []
    with open(file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            parsed_channels.append({"name": row["Name"],
                                    "gender": row["Gender"],
                                    "url": row["Channel-URL"]})
    return parsed_channels


youtubers = os.listdir(args.dataset_path)
youtubers_path = [os.path.join(args.dataset_path, youtuber) for youtuber in youtubers]
num_youtubers = len(youtubers_path)

# Obtain pretrained Haar Cascade with frontal faces:
cascade_path = args.haarcascade_path
face_cascade = cv2.CascadeClassifier(cascade_path)

for index, path in enumerate(youtubers_path):
    videos_path = os.path.join(path, 'video')
    audios_path = os.path.join(path, 'audio')
    list_videos = [filename for filename in sorted(os.listdir(videos_path)) if
                   os.path.isfile(os.path.join(videos_path, filename))]
    list_audios = [filename for filename in sorted(os.listdir(audios_path)) if
                   os.path.isfile(os.path.join(audios_path, filename)) and not filename.endswith('m4a')]
    for video, audio in zip(list_videos, list_audios):

        # Define storage directories
        cropped_audio_path = os.path.join(audios_path, audio.replace("_preprocessed.wav", "_frames"))
        cropped_faces_path = os.path.join(videos_path, video.replace(".mp4", "_cropped_frames"))
        frames_faces_path = os.path.join(videos_path, video.replace(".mp4", "_full_frames"))
        bboxes_path = os.path.join(videos_path, video.replace(".mp4", "_bboxes"))

        # Create folders if they don't exist:
        os.makedirs(cropped_audio_path, exist_ok=True)
        os.makedirs(cropped_faces_path, exist_ok=True)
        os.makedirs(frames_faces_path, exist_ok=True)
        os.makedirs(bboxes_path, exist_ok=True)

        # Get video and audio data:
        print(f'Processing videos from Youtuber {index + 1} out of {num_youtubers}')
        print('Loading video...')
        print(os.path.join(videos_path, video))
        video_capture = skvideo.io.vreader(os.path.join(videos_path, video))
        print('Finish loading video')
        wavfile_data = get_audio_samples(os.path.join(audios_path, audio))

        # Obtain video metadata:
        metadata = skvideo.io.ffprobe(os.path.join(videos_path, video))
        vmetadata = metadata["video"]
        width = int(vmetadata['@width'])
        height = int(vmetadata['@height'])
        frame_rate = vmetadata['@avg_frame_rate'].split('/')
        avg_fps = int(frame_rate[0]) / int(frame_rate[1])
        duration = vmetadata['@duration']
        duration_timestamp = vmetadata['@duration_ts']
        print(f'Video Metadata: width --> {width}, height --> {height}, Average FPS --> {avg_fps},'
              f' Duration (seconds) --> {duration}')

        # Do it only if directory is empty:
        if not os.listdir(frames_faces_path):
            print('Processing video...')
            bboxes = {}
            try:
                for idx, frame in tqdm(enumerate(video_capture)):
                    # Capture second-by-second
                    if idx % int(avg_fps) == 0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        audio_writer = wave.open(os.path.join(cropped_audio_path,
                                                              audio.replace(".wav", f"_frame_{idx}.wav")), 'wb')
                        audio_writer.setparams((1, 2, 16000, 0, 'NONE', 'Uncompressed'))
                        faces, rej_level, weights = face_cascade.detectMultiScale3(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(40, 40),
                            flags=cv2.CASCADE_SCALE_IMAGE,
                            outputRejectLevels=True
                        )

                        try:
                            max_confidence_idx = np.argmax(weights)
                            max_confidence = weights[max_confidence_idx][0]
                            print(max_confidence)
                            if max_confidence >= float(args.confidence_thresh):
                                face = faces[max_confidence_idx]
                                x, y, w, h = face
                                confidence = max_confidence
                            else:
                                # No faces detected
                                face = None
                                confidence = None
                        except (TypeError, ValueError, IndexError):
                            face = None
                            confidence = None

                        try:
                            # Save frames, cropped faces, and audios
                            if face is not None:
                                bboxes.update({idx: {'x': x, 'y': y, 'w': w, 'h': h}})
                                cropped_face = frame[y:y + h, x:x + w]  # Crop the face

                                cv2.imwrite(os.path.join(cropped_faces_path, f'cropped_face_frame_{idx}.jpg'),
                                            cropped_face)
                                cv2.imwrite(os.path.join(frames_faces_path, f'frame_{idx}.jpg'), frame)
                                cropped_wavfile = cut_audio_frame(wavfile_data, idx, avg_fps)
                                audio_writer.writeframes(cropped_wavfile.tobytes())
                                audio_writer.close()
                        except (TypeError, ValueError, cv2.error, RuntimeError) as err:
                            print(err)

                # Save bboxes:
                with open(os.path.join(bboxes_path, 'bboxes.p'), 'wb') as f:
                    pickle.dump(bboxes, f)
                check_audio_frames(frames_faces_path, cropped_audio_path)
            except RuntimeError:
                check_audio_frames(frames_faces_path, cropped_audio_path)
print('Finished')
