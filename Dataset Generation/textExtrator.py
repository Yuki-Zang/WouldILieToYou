import librosa
import numpy as np
from requests import get
import scipy as sp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor as mp

from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from datasets import load_dataset
import librosa
import datasets
import torch
from IPython.display import Audio
import moviepy.editor as mp
import os
import pandas as pd
import sox
import sys

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor as mp
from pydub import AudioSegment

# import model, feature extractor, tokenizer
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h",sampling_rate=16000)

def asr():
    dataList=[]
    '''
    extract text from the audio temp_files/temp.wav
    '''
    # Load the audio with the librosa library
    input_audio, sr = librosa.load("temp_files/temp.wav", sr=16000)

    # forward sample through model to get greedily predicted transcription ids
    input_values = feature_extractor(input_audio, return_tensors="pt").input_values
    logits = model(input_values).logits[0]
    pred_ids = torch.argmax(logits, axis=-1)

    # retrieve word stamps (analogous commands for `output_char_offsets`)
    outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
    # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
    time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate
    
    for d in outputs.word_offsets:
        dataList.append([d["word"],round(d["start_offset"] * time_offset, 2),round(d["end_offset"] * time_offset, 2)])
    
    return dataList, normalize(dataList)

#NOTE: can only be called from asr
def normalize(wordList):
    '''
    parameter: 
    wordList: list. the data list produced by the asr function

    output:
    the string of text after normalization
    '''
    text=""
    for word in wordList:
        text+=word[0]+" "

    with open('temp_files/temp.txt', 'w') as f:
        f.write(text[:-1])
        
    os.system("python normalization/recasepunc.py predict normalization/en.23000 < temp_files/temp.txt > temp_files/output.txt")

    with open('temp_files/output.txt', 'r') as f:
        normalizedText=f.read()

    return normalizedText

def write_temp_file(audioName, startTime, endTime):
    '''
    parameter:
    audioName: str. the name of the audio to be processed (without suffix)
    startTime: float. the starting time of the snippet
    endTime: float. the ending time of the snippet

    write the temp.wav file that is stored in the temp_files folder
    '''

    try:
        newAudio = AudioSegment.from_wav(f"audio/{audioName}.wav")
    except:
        print("file '{%s}' does not exist" %audioName)
        sys.exit()
    try:
        newAudio = newAudio[startTime*1000:endTime*1000]
        print(1)
    except:
        print("time segment out of range")
        sys.exit()

    #Exports to a wav file in the current path.
    newAudio.export("temp_files/temp.wav", format="wav")

def get_filled_pauses():
    '''
    uses the pratt to get the filled pauses for the audio snippet

    return:
    numFP: int. number of filled pauses in the snippet
    ifFP: 0 or 1, depending on whether there are filled pauses
    '''

    #run praat on the temp.wav file
    os.system("cp temp_files/temp.wav praat/temp.wav")
    os.system("cp temp_files/temp.wav praat/temp/temp.wav")
    os.system("/Applications/Praat.app/Contents/MacOS/Praat --run praat/syllablenucleiv3.praat ./temp None -25 2 0.3 yes English 1.00 'Save as text file' OverWriteData yes")
    os.system("cd ..")

    #read in the output file
    with open("praat/SyllableNuclei.txt", "r") as file:
        contents = file.read().split(", ")
    numFP = int(contents[-2])
    ifFP = 0 if numFP==0 else 1
    return numFP, ifFP

