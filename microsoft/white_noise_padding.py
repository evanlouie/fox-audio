from pydub import AudioSegment
from pydub.generators import WhiteNoise
import wave
import sys
import os
import math
import numpy as np
import shutil
import subprocess
import argparse
import glob
import multiprocessing


def wav_length(filename):
    wav = wave.open(filename, "r")
    frames = wav.getnframes()
    rate = wav.getframerate()
    duration = frames / float(rate)
    return duration


def combinations(lst, target, with_replacement=False):
    def _a(idx, l, r, t, w):
        if t == sum(l):
            r.append(l)
        elif t < sum(l):
            return
        for u in range(idx, len(lst)):
            _a(u if w else (u + 1), l + [lst[u]], r, t, w)
        return r

    return _a(0, [], [], target, with_replacement)


def padding(wav, white_noise_duration):
    # print("WAV FILE: " + wav)
    for x in white_noise_duration:
        if x == 0:
            wav_files = []
            padded_fname = wav.rsplit(".", 1)[0]
            # print("PADDED NAME: " + padded_fname)
            silence_duration = max(white_noise_duration)
            # print(padded_fname+"_whitenoise.wav")

            # convert sampling rate, bits per sample, audio channel
            subprocess.call(
                [
                    "ffmpeg",
                    "-i",
                    wav,
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    padded_fname + "_converted.wav",
                    "-y",
                ]
            )

            # white noise duration should be a list e.g [0,1]
            # generate white noise wav file
            wn = WhiteNoise().to_audio_segment(duration=silence_duration * 1000)
            wn.export(
                padded_fname + "_whitenoise.wav",
                format="wav",
                parameters=["-ar", "16000"],
            )

            # stitch white noise wav file to specific audio wav file
            # before
            new_wav_before = AudioSegment.from_wav(
                padded_fname + "_whitenoise.wav"
            ) + AudioSegment.from_wav(padded_fname + "_converted.wav")
            new_wav_before.export(
                padded_fname
                + "_padded"
                + "_"
                + str(white_noise_duration[1])
                + "_"
                + str(white_noise_duration[0])
                + ".wav",
                format="wav",
                parameters=["-ar", "16000"],
            )

            # after
            new_wav_after = AudioSegment.from_wav(
                padded_fname + "_converted.wav"
            ) + AudioSegment.from_wav(padded_fname + "_whitenoise.wav")
            new_wav_after.export(
                padded_fname
                + "_padded"
                + "_"
                + str(white_noise_duration[0])
                + "_"
                + str(white_noise_duration[1])
                + ".wav",
                format="wav",
                parameters=["-ar", "16000"],
            )

            # remove white noise wav file
            os.remove(padded_fname + "_whitenoise.wav")
            os.remove(padded_fname + "_converted.wav")
            wav_files.append(
                padded_fname
                + "_padded"
                + "_"
                + str(white_noise_duration[1])
                + "_"
                + str(white_noise_duration[0])
                + ".wav"
            )
            wav_files.append(
                padded_fname
                + "_padded"
                + "_"
                + str(white_noise_duration[0])
                + "_"
                + str(white_noise_duration[1])
                + ".wav"
            )
            break
        else:
            wav_files = []
            padded_fname = (wav.rsplit(".", 1)[0]).split("/")[-1]
            # print("PADDED FILENAME: " + padded_fname)
            path = (wav.rsplit(".", 1)[0]).rsplit("/", 1)[0]
            # print("PATH: "+ path)
            fn = (wav.rsplit(".", 1)[0]).rsplit("/", 1)[1]
            # print("FILENAME: " + fn)

            # white noise duration should be a list e.g [0,1]
            # generate white noise wav file
            # wn_0 = AudioSegment.silent(duration=white_noise_duration[0] * 1000)
            wn_0 = WhiteNoise().to_audio_segment(
                duration=white_noise_duration[0] * 1000
            )
            wn_0.export(
                wav + "_whitenoise_0.wav", format="wav", parameters=["-ar", "16000"]
            )

            # wn_1 = AudioSegment.silent(duration=white_noise_duration[1] * 1000)
            wn_1 = WhiteNoise().to_audio_segment(
                duration=white_noise_duration[1] * 1000
            )
            wn_1.export(
                wav + "_whitenoise_1.wav", format="wav", parameters=["-ar", "16000"]
            )

            # stitch white noise wav file to specific audio wav file
            new_wav = (
                AudioSegment.from_wav(wav + "_whitenoise_0.wav")
                + AudioSegment.from_wav(wav)
                + AudioSegment.from_wav(wav + "_whitenoise_1.wav")
            )
            new_wav.export(
                path
                + "/"
                + padded_fname
                + "_padded"
                + "_"
                + str(white_noise_duration[0])
                + "_"
                + str(white_noise_duration[1])
                + ".wav",
                format="wav",
                parameters=["-ar", "16000"],
            )

            # after
            new_wav_reverse = (
                AudioSegment.from_wav(wav + "_whitenoise_1.wav")
                + AudioSegment.from_wav(wav)
                + AudioSegment.from_wav(wav + "_whitenoise_0.wav")
            )
            new_wav_reverse.export(
                path
                + "/"
                + padded_fname
                + "_padded"
                + "_"
                + str(white_noise_duration[1])
                + "_"
                + str(white_noise_duration[0])
                + ".wav",
                format="wav",
                parameters=["-ar", "16000"],
            )

            # remove white noise wav file
            os.remove(wav + "_whitenoise_0.wav")
            os.remove(wav + "_whitenoise_1.wav")

            wav_files.append(
                path
                + "/"
                + padded_fname
                + "_padded"
                + "_"
                + str(white_noise_duration[0])
                + "_"
                + str(white_noise_duration[1])
                + ".wav"
            )
            wav_files.append(
                path
                + "/"
                + padded_fname
                + "_padded"
                + "_"
                + str(white_noise_duration[1])
                + "_"
                + str(white_noise_duration[0])
                + ".wav"
            )

            # If adding to one folder, specify the path of folder!
            # new_wav.export("output_/"+fn+"_padded"+"_"+str(white_noise_duration[0])+"_"+str(white_noise_duration[1])+".wav", format="wav", parameters=["-ar", "16000"])
            # new_wav_reverse.export("output_/"+fn+"_padded"+"_"+str(white_noise_duration[1])+"_"+str(white_noise_duration[0])+".wav", format="wav", parameters=["-ar", "16000"])

            break
    return wav_files


def padding_output(f):
    try:
        fname = f
        # fname = target_directory + "/" + f
        n = wav_length(fname)
        print("length of wav file: " + str(n) + " seconds")
        float(n)

        # round the wav up to the nearest integer
        integ = math.ceil(n)
        if integ >= 10:
            raise ValueError(
                "Clip: "
                + fname
                + " is more than 10 or more seconds long. Cannot pad file."
            )
        else:
            silence_remain = integ - n
            print("white noise remainder: " + str(silence_remain) + " seconds")
            float(silence_remain)
            rand = np.random.uniform(0, silence_remain)
            rand_remain = silence_remain - rand
            random = [rand, rand_remain]
            print(
                "random white noise padding: "
                + str(random[0])
                + " and "
                + str(random[1])
            )
            float(random[0]) and float(random[1])

            if silence_remain == 0:
                remainder = 10 - round(n)
                int(remainder)
                lst = list(range(0, 10))
                comb = combinations(lst, remainder)
                comb_new = []
                for i in comb:
                    if len(i) == 2:
                        comb_new.append(i)
                # print(fname)
                folder_name = (fname.split("/")[-1]).split(".")[0]
                # print(folder_name)
                path = "output/"
                if not os.path.exists(path):
                    os.mkdir(path)

                shutil.copy(fname, path)
                destination = path + "/" + folder_name + ".wav"
                os.rename(path + "/" + fname.split("/")[-1], destination)

                for combination in comb_new:
                    # 10 second padding
                    padding(destination, combination)
            else:
                # pad audio file
                # rounded_wav = padding(sys.argv[1], [0, silence_remain])
                rounded_wav = padding(fname, random)
                print(rounded_wav)
                rounded_wav_length_0 = wav_length(rounded_wav[0])
                # print("Length of rounded wav file: " + str(rounded_wav_length_0) + " seconds")
                # float(rounded_wav_length_0)

                rounded_wav_length_1 = wav_length(rounded_wav[1])
                # print("Length of rounded wav file: " + str(rounded_wav_length_1) + " seconds")
                # float(rounded_wav_length_1)

                # determine all combinations of a + b = remainder
                remainder = 10 - round(rounded_wav_length_1)
                # print("remainder value: " + str(remainder))
                int(remainder)
                lst = list(range(0, 10))
                comb = combinations(lst, remainder)
                comb_new = []
                for i in comb:
                    if len(i) == 2:
                        comb_new.append(i)
                # print(comb_new)

                # rename padded wav file
                # copy padded wav file in output folder as new file
                folder_name = (fname.rsplit(".", 1)[0]).split("/")[-1]
                # print("FOLDER_NAME: " + folder_name)
                path = "output/"
                if not os.path.exists(path):
                    os.mkdir(path)

                shutil.copy(rounded_wav[1], path)
                # destination = path+"/"+ folder_name+".wav"
                destination = path + folder_name + ".wav"
                # print("DESTINATION: " + destination)
                os.rename(path + "/" + rounded_wav[1].split("/")[-1], destination)

                for combination in comb_new:
                    # 10 second padding
                    padding(destination, combination)

                # remove in wav folder
                for file in rounded_wav:
                    os.remove(file)
        os.remove(destination)
    except Exception:
        print("Error on: " + f)


if __name__ == "__main__":

    def parse_args():
        """Parse out the required arguments to run as script"""
        parser = argparse.ArgumentParser(description="Add white padding to WAV files")
        parser.add_argument("target_dir", help="Target directory of WAV files")
        parser.add_argument(
            "--output_dir",
            help="Output path for padded WAV files. Will be created if does not exist (default: output)",
            default="output",
        )
        return parser.parse_args()

    # Parse user input
    args = parse_args()
    output_dir = args.output_dir
    target_dir = args.target_dir

    # make directory for output files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.isdir(os.path.join(os.path.curdir, target_dir)):
        # Create glob patterns for possible filename extensions
        wav_glob_patterns = [
            os.path.join(target_dir, "**/*" + ext)
            for ext in [".wav", ".wave", ".WAV", ".WAVE"]
        ]
        # Flatten out the globs with `sum`
        wav_filepaths = sum(map(glob.glob, wav_glob_patterns), [])
        # Process in parallel
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.map(padding_output, wav_filepaths)
    else:
        padding_output(sys.argv[1])
