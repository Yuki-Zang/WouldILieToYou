Filenames and functions:

Video Prep
------------------------------------------------------------------------
(Note: The videos we got had titles including the episode id and guest name. If yours don't have this information, changes are needed with the following files.)
video2audio.ipynb: convert videos (mp4) to audios (mp3) and meanwhile extract the guest name info. 
rename_video.ipynb: rename each video into sXXeXX.mp4.
compress_video.ipynb: reduce the file size of videos (240MB -> 40MB) for annotation purpose.
*cut_video.ipynb: extract a small section of audio (wav) from the original video (mp4).

Dataset Generation
------------------------------------------------------------------------
**Diartization:**

The diart library reads in wav file and output the rttm file which detects speaker changes. The exact second of a speaker change is useful to delimit the utterances. You may refer to https://pypi.org/project/diart/#description if you want to diartize new videos. We also provide all rttm files for 14 seasons of WILTY in the folder "rttm".

**Section Delimitation:**

read_rttm.ipynb: take in rttm files from diartization and generate time segmentation files (csv)
delimit_sections.ipynb: take the timeSeg info and detection sounds (see in folder `sounds') to delimit the sections and utterances. Return jsonl files, which will be used in prodigy annotation. You can set the number of beaps(answer-revealing sounds) to be detected when running the file. The number varies for different episodes, but 6 is a good start.
*wordRecognizer.ipynb: generate files to record each spoken word and its start, end time from a video
*diartSegGenerator.ipynb: generate time nodes for speaker-role changes and do speech recognition on audio files

**Annotation:**

make sure you are in a directory that has these folders:
1) json - inside this will be a folder s01, s02, etc
2) videoAnnotate - inside this will be all the small videos

I. first do:
python preprocess/addb64.py json/s<season> s<season>.jsonl

II. then do:
python -m prodigy audio.manual s<season>final s<season>.jsonl --loader jsonl --label s1,s2,s3,s4,s5,srob,sdavid,slee,sangus,start,end_true,end_false

III. then, to label:
* scroll along til you get to a speaker turn that starts a segment - when the speaker picks up the card - and mark it "start"
* scroll along til you get to the speaker's first telling of the story (reading the card, or the possession story) and mark it s1,s2,s3, slee etc - the speaker
* scroll along til the end of the segment, fix the end boundary so it ends at the end of the segment (after the true/false has shown), and make sure you use end_true or end_false
* if it was a possession segment, in the csv, in the 4th column, write the number of the segment (starting from 1, not 0!)
* and so on til the end of the episode
* and then _save your annotations_!!
then to back up the annotations:
python -m prodigy db-out s<season>final > s<season>out.jsonl
and upload that jsonl to google drive

you will probably want to make a prodigy.json file and put it wherever you will start prodigy from, contents:
{
  "audio_autoplay": false,
  "audio_rate": 1.5,
  "audio_max_zoom": 500
}

This makes it play twice as fast, and lets you zoom in on the audio waveform if you really need to.

Same as above, you may reannotate to extend beyond what we did, or we provide the annotation in folder `jsonlOut'.

**Feature Extraction:**

featuerGenerator.ipynb: generate the dataset
featureExtractor.py: a library to be imported in featureGenerator.ipynb
modelTrainer: train basic models on the dataset

**Visual Feature Extraction:**

See multimodal/visual_features.ipynb for procedure.
Expects video segments as a list of frames (10 frames/second).

Modeling
------------------------------------------------------------------------
**Speaker_label:**

We provide the human labels (see folder "speaker_label"). "1" means True, "0" means False, and "-1" indicates that the speaker is noncommittal.

**Models**

  * See multimodal/model.ipynb for model training procedure based on sklearn models.
  * See multimodal/BiLSTM.ipynb for model training procedure based on BiLSTM.
  
For transformer based feature fusion using Perceiver
  * Use multimodal/viper_train.py for training [Perceiver](https://huggingface.co/docs/transformers/model_doc/perceiver) model using [RAVDESS data].(https://zenodo.org/records/1188976)
  * Use multimodal/finetune_wilty.py for finetuning trained model using would I lie to you dataset.
  
  Original Viper code is available [here](https://github.com/VaianiLorenzo/ViPER/tree/main)
