import argparse
import wave
import audioop
import subprocess as sp
import os
import unicodedata

parser = argparse.ArgumentParser()
parser.add_argument("-dataset_path", required=True,
                    help="Parent directory where the dataset is stored")
args = parser.parse_args()
errors = 0


def format_filename(filename):
    try:
        s = ''.join((c for c in unicodedata.normalize('NFD', filename) if unicodedata.category(c) != 'Mn'))
        return s
    except (UnicodeEncodeError, UnicodeDecodeError):
        return filename


def aac2wav(input_file, target_file, errors):
    try:
        command = 'ffmpeg -i "{}" -vn "{}"'.format(input_file, target_file)
        print(command)
        sp.check_call(command, shell=True)
        return errors
    except (sp.CalledProcessError, IOError) as e:
        print(f"Error converting {input_file} to {target_file}: {e}")
        errors += 1
        return errors


youtubers = os.listdir(args.dataset_path)
youtubers_dataset = [os.path.join(args.dataset_path, youtuber, 'audio') for youtuber in youtubers]

for youtuber_audio_path in youtubers_dataset:
    audio_files = os.listdir(youtuber_audio_path)
    for aac_audio_fname in audio_files:
        original_file = os.path.join(youtuber_audio_path, aac_audio_fname)
        new_fname = format_filename(aac_audio_fname.replace(" ", "").replace("'", "").replace('"', '').replace('(', '')
                                    .replace(')', '').replace('#', '').replace('&', '').replace(';', '').replace('!', '')
                                    .replace(',', '').replace('$', ''))
        new_file = os.path.join(youtuber_audio_path, new_fname)
        os.rename(original_file, new_file)
        aac_audio_fname = new_fname
        print('Processing {} file'.format(aac_audio_fname))
        # Convert from AAC to WAV:
        if not aac_audio_fname.endswith(".wav"):
            wav_audio_fname = aac_audio_fname.replace(".m4a", ".wav")
        else:
            print('Was already a wav file!!!')
            wav_audio_fname = aac_audio_fname  # It is not AAC but WAV

        if not os.path.exists(os.path.join(youtuber_audio_path, wav_audio_fname)):
            errors = aac2wav(os.path.join(youtuber_audio_path, aac_audio_fname),
                             os.path.join(youtuber_audio_path, wav_audio_fname), errors)

        if not wav_audio_fname.endswith("_preprocessed.wav"):
            output_file = wav_audio_fname.replace(".wav", "_preprocessed.wav")
        else:
            output_file = wav_audio_fname

        if not os.path.exists(os.path.join(youtuber_audio_path, output_file)):
            # Read audio and obtain metadata:
            try:
                audio_reader = wave.open(os.path.join(youtuber_audio_path, wav_audio_fname), 'rb')
                nchannels, sampwidth, framerate, nframes, comptype, compname = audio_reader.getparams()
                data = audio_reader.readframes(nframes)
                # Downsample to 16kHz, 16 bits, mono:
                converted = audioop.ratecv(data, sampwidth, nchannels, framerate, 16000, None)
                if sampwidth != 2:
                    print('Converting sample width from {} to 2'.format(sampwidth))
                    converted = (audioop.lin2lin(converted[0], sampwidth, 2), converted[1])
                if nchannels != 1:
                    converted = (audioop.tomono(converted[0], 2, 1, 0), converted[1])
                # Write output audio file:
                audio_writer = wave.open(os.path.join(youtuber_audio_path, output_file), 'wb')
                audio_writer.setparams((1, 2, 16000, 0, 'NONE', 'Uncompressed'))
                audio_writer.writeframes(converted[0])
                audio_reader.close()
                audio_writer.close()
            except Exception as e:
                print('Failed to process wav file: {}'.format(e))
                errors += 1

print('Found {} errors'.format(errors))
