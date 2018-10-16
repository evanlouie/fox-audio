from pydub import AudioSegment
import wave
import sys
import os
import math
import numpy as np
import shutil
from pydub.generators import WhiteNoise 
import subprocess

def wav_length(fname):
    wav = wave.open(fname,'r')
    frames = wav.getnframes()
    rate = wav.getframerate()
    duration = frames / float(rate)
    return(duration)

def combinations(lst, target, with_replacement=False):
    def _a(idx, l, r, t, w):
        if t == sum(l): r.append(l)
        elif t < sum(l): return
        for u in range(idx, len(lst)):
            _a(u if w else (u + 1), l + [lst[u]], r, t, w)
        return r
    return _a(0, [], [], target, with_replacement)

def padding(wav, white_noise_duration):
    #print("WAV FILE: " + wav)
    for x in white_noise_duration:
        if x == 0:
            wav_files = []
            padded_fname = (wav.rsplit('.', 1)[0])
            #print("PADDED NAME: " + padded_fname)
            silence_duration = max(white_noise_duration)
            #print(padded_fname+"_whitenoise.wav")

            # convert sampling rate, bits per sample, audio channel
            subprocess.call(['ffmpeg', '-i', wav, '-ar', "44100", '-ac', "2", padded_fname+"_converted.wav", '-y'])

            # white noise duration should be a list e.g [0,1]
            # generate white noise wav file
            wn = WhiteNoise().to_audio_segment(duration=silence_duration * 1000)
            wn.export(padded_fname+"_whitenoise.wav",format="wav", parameters=["-ar", "16000"])

            # stitch white noise wav file to specific audio wav file
            # before
            new_wav_before = AudioSegment.from_wav(padded_fname+"_whitenoise.wav") + AudioSegment.from_wav(padded_fname+"_converted.wav")
            new_wav_before.export(padded_fname+"_padded"+"_"+str(white_noise_duration[1])+"_"+str(white_noise_duration[0])+".wav", format="wav", parameters=["-ar", "16000"])

            # after
            new_wav_after = AudioSegment.from_wav(padded_fname+"_converted.wav") + AudioSegment.from_wav(padded_fname+"_whitenoise.wav")
            new_wav_after.export(padded_fname+"_padded"+"_"+str(white_noise_duration[0])+"_"+str(white_noise_duration[1])+".wav", format="wav", parameters=["-ar", "16000"])

            # remove white noise wav file
            os.remove(padded_fname+"_whitenoise.wav")
            os.remove(padded_fname+"_converted.wav")
            wav_files.append(padded_fname+"_padded"+"_"+str(white_noise_duration[1])+"_"+str(white_noise_duration[0])+".wav")
            wav_files.append(padded_fname+"_padded"+"_"+str(white_noise_duration[0])+"_"+str(white_noise_duration[1])+".wav")
            break
        else:
            wav_files = []
            padded_fname = (wav.rsplit('.', 1)[0]).split('/')[-1]
            #print(padded_fname)
            path = (wav.rsplit('.', 1)[0]).rsplit('/',1)[0]
            #print("PATH: "+ path)

            # white noise duration should be a list e.g [0,1]
            # generate white noise wav file
            #wn_0 = AudioSegment.silent(duration=white_noise_duration[0] * 1000) 
            wn_0 = WhiteNoise().to_audio_segment(duration=white_noise_duration[0] * 1000)
            wn_0.export(wav+"_whitenoise_0.wav",format="wav", parameters=["-ar", "16000"])

            #wn_1 = AudioSegment.silent(duration=white_noise_duration[1] * 1000) 
            wn_1 = WhiteNoise().to_audio_segment(duration=white_noise_duration[1] * 1000)
            wn_1.export(wav+"_whitenoise_1.wav",format="wav", parameters=["-ar", "16000"])

            # stitch white noise wav file to specific audio wav file
            new_wav = AudioSegment.from_wav(wav+"_whitenoise_0.wav") + AudioSegment.from_wav(wav) + AudioSegment.from_wav(wav+"_whitenoise_1.wav")
            new_wav.export(path+"/"+padded_fname+"_padded.wav"+"_"+str(white_noise_duration[0])+"_"+str(white_noise_duration[1])+".wav", format="wav", parameters=["-ar", "16000"])

            # after
            new_wav_reverse = AudioSegment.from_wav(wav+"_whitenoise_1.wav") + AudioSegment.from_wav(wav) + AudioSegment.from_wav(wav+"_whitenoise_0.wav")
            new_wav_reverse.export(path+"/"+padded_fname+"_padded"+"_"+str(white_noise_duration[1])+"_"+str(white_noise_duration[0])+".wav", format="wav", parameters=["-ar", "16000"])

            # remove white noise wav file
            os.remove(wav+"_whitenoise_0.wav")
            os.remove(wav+"_whitenoise_1.wav")
            
            wav_files.append(padded_fname+"_padded.wav"+"_"+str(white_noise_duration[0])+"_"+str(white_noise_duration[1])+".wav")
            wav_files.append(padded_fname+"_padded.wav"+"_"+str(white_noise_duration[1])+"_"+str(white_noise_duration[0])+".wav")
            break
    return wav_files

# make directory for output files
newpath = r'output' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

# take in wav file and calculate duration, n
fname = sys.argv[1]
n = wav_length(fname)
print("length of wav file: " + str(n) + " seconds")
float(n)

# round the wav up to the nearest integer
integ = math.ceil(n)
if integ >= 10:
    raise ValueError('A clip is 10 second or more long. Cannot add padding.')
else:
    silence_remain = integ - n
    print("white noise remainder: " + str(silence_remain) + " seconds")
    float(silence_remain)

    if silence_remain == 0:
        remainder = 10 - round(n)
        int(remainder)
        lst = list(range(0, 10))
        comb = combinations(lst, remainder)
        comb_new = []
        for i in comb:
            if len(i) == 2:
                comb_new.append(i)
        #print(fname)
        folder_name = (fname.split("/")[-1]).split(".")[0]
        #print(folder_name)
        path = "output/"+folder_name
        if not os.path.exists(path): 
            os.mkdir(path)

        shutil.copy(fname, path)
        destination = path+"/"+ folder_name+".wav"
        os.rename(path+"/"+fname.split("/")[-1], destination)

        for combination in comb_new:
            # 10 second padding
            padding(destination,combination)
    else: 
        # pad audio file
        rounded_wav = padding(sys.argv[1], [0, silence_remain])
        print(rounded_wav)
        rounded_wav_length_0 = wav_length(rounded_wav[0])
        print("Length of rounded wav file: " + str(rounded_wav_length_0) + " seconds")
        float(rounded_wav_length_0)

        rounded_wav_length_1 = wav_length(rounded_wav[1])
        #print("Length of rounded wav file: " + str(rounded_wav_length_1) + " seconds")
        #float(rounded_wav_length_1)

        # determine all combinations of a + b = remainder
        remainder = 10 - round(rounded_wav_length_1)
        #print("remainder value: " + str(remainder))
        int(remainder)
        lst = list(range(0, 10))
        comb = combinations(lst, remainder)
        comb_new = []
        for i in comb:
            if len(i) == 2:
                comb_new.append(i)
        #print(comb_new)

        # rename padded wav file
        # copy padded wav file in output folder as new file
        folder_name = (fname.rsplit('.', 1)[0]).split('/')[-1]
        #print("FOLDER_NAME: " + folder_name)
        path = "output/"+folder_name
        if not os.path.exists(path): 
            os.mkdir(path)

        shutil.copy(rounded_wav[1], path)
        destination = path+"/"+ folder_name+".wav"
        #print("DESTINATION: " + destination)
        os.rename(path+"/"+rounded_wav[1].split("/")[-1], destination)

        for combination in comb_new:
            # 10 second padding
            padding(destination,combination)

        # remove in wav folder
        for file in rounded_wav:
            os.remove(file)