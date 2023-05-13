

## ==> GUI FILE

from PyQt5 import QtCore
from PyQt5.QtCore import QPropertyAnimation, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QTableWidgetItem
import PyQt5.QtCore  
from PyQt5.QtWidgets import QMessageBox
import os
from datetime import datetime, timedelta
import mysql.connector
from PySide2.QtWidgets import *
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from main import MainWindow
import numpy   as np
import os
from utils import *
from pathlib import Path
from pydub import AudioSegment
from resemblyzer import preprocess_wav, VoiceEncoder
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances,silhouette_score
import numpy as np
import os
import wave
from vosk import Model, KaldiRecognizer, SetLogLevel
import json
from resemblyzer import sampling_rate
import pyaudio
import ffmpeg
# pip install webrtcvad-wheels
from scipy.io import wavfile
from main import * 
from sklearn.metrics.pairwise import cosine_similarity
import math
import time as timeD
from pydub.effects import low_pass_filter, high_pass_filter
import noisereduce as nr
from sklearn.cluster import AgglomerativeClustering
from webrtcvad import Vad
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score



SetLogLevel(-1)

class SpeechThread(QThread):
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui


    # Add Voice Activity Detection (VAD)
    def filter_silence(self,audio_data, frame_duration=30):
        vad = Vad(3)
        samples_per_frame = int(16000 * frame_duration / 1000)
        n_frames = len(audio_data) // samples_per_frame

        voiced_audio = []
        for i in range(n_frames):
            frame = audio_data[i * samples_per_frame:(i + 1) * samples_per_frame]
            if vad.is_speech(frame.tobytes(), 16000):
                voiced_audio.extend(frame)
        return np.array(voiced_audio)

    

    def convert_to_mono_wav(self,filename, output_dir):
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension not in ['.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg']:
            raise ValueError("Unsupported file format")

        sound = AudioSegment.from_file(filename, format=file_extension[1:])
        sound = sound.set_channels(1)
        new_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0] + ".wav")
        sound.export(new_filename, format="wav")
        sampling_rate, wav = wavfile.read(new_filename)

        return new_filename, sampling_rate, wav
        
    def merge_short_segments(self, labelling, min_duration=1):
            merged_labelling = []
            prev_speaker, prev_interval = labelling[0]
            prev_start, prev_end = [float(x) for x in prev_interval.strip('()').split(' - ')]

            for speaker, interval in labelling[1:]:
                start, end = [float(x) for x in interval.strip('()').split(' - ')]
                if speaker == prev_speaker and (start - prev_end) < min_duration:
                    prev_end = end
                else:
                    merged_labelling.append([prev_speaker, f"({prev_start:.2f} - {prev_end:.2f})"])
                    prev_speaker, prev_start, prev_end = speaker, start, end

            merged_labelling.append([prev_speaker, f"({prev_start:.2f} - {prev_end:.2f})"])
            return merged_labelling
    
    def cosine_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))



    def speaker_verification_and_merge(self, clustering, cluster_mean_embeddings, similarity_threshold=0.8):
            n_clusters = len(cluster_mean_embeddings)
            merged_labels = clustering.labels_.copy()
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    similarity = self.cosine_similarity(cluster_mean_embeddings[i], cluster_mean_embeddings[j])
                    if similarity > similarity_threshold:
                        merged_labels[merged_labels == j] = i

            return merged_labels
    
    def split_voiced_audio_segments(self, voiced_audio, wav_splits, labels):
        audio_segments = []
        for i, label in enumerate(labels):
            start = wav_splits[i].start
            stop = wav_splits[i].stop
            audio_segment = voiced_audio[start:stop]
            audio_segments.append((label, audio_segment))
        return audio_segments
    
    def calculate_buffer_size(self,confidence):
        MIN_BUFFER_SIZE = 0.1
        MAX_BUFFER_SIZE = 0.5
        BUFFER_SCALE = 0.4
        
        buffer_size = MIN_BUFFER_SIZE + (1 - confidence) * BUFFER_SCALE
        return min(buffer_size, MAX_BUFFER_SIZE)
    
    def calculate_speaker_change_confidence(self,cluster_mean_embeddings, current_label, prev_label):
        cosine_similarity = np.dot(cluster_mean_embeddings[current_label], cluster_mean_embeddings[prev_label]) / (np.linalg.norm(cluster_mean_embeddings[current_label]) * np.linalg.norm(cluster_mean_embeddings[prev_label]))
        return cosine_similarity

    



    def run(self):

        #get the selected model 
        selected_model = self.ui.selectModel.currentText()

        if selected_model == "English 1" :
            trans_model = "trained-model"
        elif selected_model == "Filipino" :
            trans_model = "vosk-model-tl-ph-generic-0.6"

        
        # Get the file
        filename = self.ui.uploadFilename.text()
        output_dir = 'uploads'
        os.makedirs(output_dir, exist_ok=True)

        filename, sampling_rate, wav = self.convert_to_mono_wav(filename, output_dir)


       
        BATCH_SIZE = 100

        # Transcription
        encoder = VoiceEncoder()

        # Load the audio file
        audio = AudioSegment.from_file(filename, format="wav")

        # Resample the audio data to 16 kHz
        audio = audio.set_frame_rate(16000)

        # Apply a bandpass filter to remove low and high frequencies
        filtered_audio = audio.high_pass_filter(100).low_pass_filter(8000)

        # Convert the filtered audio to a numpy array
        audio_data = np.array(filtered_audio.get_array_of_samples())

        # Apply noise reduction to remove background noise
        noise_profile = nr.reduce_noise(y=audio_data, sr=16000)
        audio_data = np.array(noise_profile)

        # Apply VAD to remove silence
        voiced_audio = self.filter_silence(audio_data)



        # Normalize the audio signal to [-1, 1]
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 0:
            audio_data = audio_data / max_amplitude

        # Preprocess the audio data and extract continuous embeddings
        preprocessed_wav = preprocess_wav(audio_data, 16000)
        _, cont_embeds, wav_splits = encoder.embed_utterance(preprocessed_wav, return_partials=True)
        cont_embeds = np.array(cont_embeds).astype(np.float64)

        # Standardize the embeddings
        scaler = StandardScaler()
        cont_embeds_std = scaler.fit_transform(cont_embeds)

        # Perform speaker clustering
        max_clusters = 5
        score_avgs = []

        for n_clusters in range(2, max_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
            labels = clustering.fit_predict(cont_embeds_std)

            silhouette_avg = silhouette_score(cont_embeds_std, labels)
            calinski_harabasz_avg = calinski_harabasz_score(cont_embeds_std, labels)
            davies_bouldin_avg = davies_bouldin_score(cont_embeds_std, labels)

            combined_score = silhouette_avg + calinski_harabasz_avg - davies_bouldin_avg
            score_avgs.append(combined_score)

        optimal_clusters = np.argmax(score_avgs) + 2
        optimal_clusters = min(optimal_clusters, max_clusters)

        clustering = AgglomerativeClustering(n_clusters=optimal_clusters, metric='euclidean', linkage='ward')
        labels = clustering.fit_predict(cont_embeds_std)

        # Perform speaker verification and merge similar clusters
        cluster_mean_embeddings = np.array([np.mean(cont_embeds_std[labels == i], axis=0) for i in range(optimal_clusters)])
        merged_labels = self.speaker_verification_and_merge(clustering, cluster_mean_embeddings)
        labels = merged_labels

        # Label the speaker segments based on the clustering result
        times = np.array([(s.start + s.stop) / 2 for s in wav_splits])
        prev_label = labels[0]
        labelling = [["Speaker 1", f"({wav_splits[0].start:.2f} - {times[0]:.2f})"]]

        for i in range(1, len(times)):
            if labels[i] != prev_label or times[i] - times[i-1] >= 20:
                # Speaker change detected or time elapsed since the last label is greater than 20 seconds
                start_time = times[i-1]
                end_time = times[i]
                speaker_number = prev_label + 1
                temp = ["Speaker " + str(speaker_number), f"({start_time:.2f} - {end_time:.2f})"]
                labelling.append(temp)
            prev_label = labels[i]

        # Label the last speaker segment
        start_time = times[-1]
        end_time = wav_splits[-1].stop
        speaker_number = prev_label + 1
        temp = ["Speaker " + str(speaker_number), f"({start_time:.2f} - {end_time:.2f})"]
        labelling.append(temp)

        labelling = self.merge_short_segments(labelling)
        labelling = sorted(labelling, key=lambda x: float(x[1].split(" - ")[0][1:]))





        self.ui.consoleLog.append("Speaker diarization completed!")


        model = Model("model/" + trans_model)
        wf = wave.open(filename, "rb")

        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)


        # Transcribe the audio
        transcription = []
        total_frames = wf.getnframes()
        frame_rate = wf.getframerate()
        chunk_size = frame_rate  # Read in chunks of 1 second
        frames_read = 0

        while frames_read < total_frames:
                remaining_frames = total_frames - frames_read
                frames_to_read = min(chunk_size, remaining_frames)
                data = wf.readframes(frames_to_read)
                frames_read += frames_to_read
                progress = int(frames_read / total_frames * 100)
                print(f"Transcribing... {progress}% complete", end="\r")
                self.ui.consoleLog.append(f"Transcribing... {progress}% complete")
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    # Convert json output to dict
                    result_dict = json.loads(rec.Result())
                    # Check if the "result" key is present in the dict
                    if "result" in result_dict:
                        # Extract start and end timestamps and text values
                        start = result_dict["result"][0]["start"]
                        end = result_dict["result"][-1]["end"]
                        text = " ".join([word["word"] for word in result_dict["result"]])
                        # Append dictionary to transcription list
                        transcription.append({"start": start, "end": end, "text": text})
                    else:
                        # Handle the case when the "result" key is not present in the dict
                        print("No result found in transcription")

            # Get the final result after the loop
        result_dict = json.loads(rec.FinalResult())
        if "result" in result_dict:
            start = result_dict["result"][0]["start"]
            end = result_dict["result"][-1]["end"]
            text = " ".join([word["word"] for word in result_dict["result"]])
            transcription.append({"start": start, "end": end, "text": text})
        else:
            print("No result found in the final transcription")

        # Sort transcription list by start time
        transcription = sorted(transcription, key=lambda x: x["start"])

        print(transcription)

        # Output the results in the format Speaker Number (Timestamp) : Transcription
        prev_end = 0
        for l, t in zip(labelling, transcription):
            start_time = str(timedelta(seconds=int(t["start"]))).split(".")[0]
            end_time = str(timedelta(seconds=int(t["end"]))).split(".")[0]
            if t["start"] >= prev_end:
                trans = ("%s %s : \n%s \n" % (l[0], f"({start_time} - {end_time})", t["text"]))
            else:
                start_time = str(timedelta(seconds=int(prev_end))).split(".")[0]
                end_time = str(timedelta(seconds=int(t["end"]))).split(".")[0]
                trans = ("%s %s :\n  %s \n" % (l[0], f"({start_time} - {end_time})", t["text"]))
            print(trans)
            self.ui.outputText.append(trans)

            timeD.sleep(1)

            prev_end = t["end"]



            
class SpeechLive(QThread):
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.stop_recording = False


    def run(self):
       # Initialize PyAudio
        p = pyaudio.PyAudio()
        # Get list of available audio devices
        device_index = self.ui.selectMicrophone.currentIndex()
        print("Using device", p.get_device_info_by_index(device_index).get('name'))
        device_index = device_index
        
        # Open audio stream from selected microphone
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000, input_device_index=device_index)

        # Initialize Vosk model and recognizer
        # model = Model("vosk-model-en-us-aspire-0.2")
        selected_model = self.ui.selectLiveModel.currentText()

        if selected_model == "English 1" :
            trans_model = "trained-model"
        elif selected_model == "Filipino" :
            trans_model = "vosk-model-tl-ph-generic-0.6"


        model = Model("model/"+trans_model)
        rec = KaldiRecognizer(model, 16000)

        # Print message to indicate recording has started
        print("Recording started")
        self.ui.btnStopLive.setEnabled(True)
        self.ui.btnStopLive.setStyleSheet("background-color: red;")

        self.ui.btnStartLive.setEnabled(False)
        self.ui.btnStartLive.setStyleSheet("background-color: #024d00;")


        # Transcription list to store results
        transcription = []
        transcription_paragraphs = {}
        while not self.stop_recording:
            data = stream.read(16000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                # Convert json output to dict
                result_dict = json.loads(rec.Result())
                # Extract text value from the dict
                result_text = result_dict.get("text", "")
                # Print the result live on the console
                # words_spoken = (" "+result_text, end='', flush=True)
              
                # Append the result to the transcription list
                transcription.append(result_text)
                self.ui.outputText_Live.append(result_text)
            if self.stop_recording:
                break
        stream.stop_stream()
        stream.close()
        p.terminate()
              
    
    def stopRecording(self):
        print("Recording Stopped")
        self.stop_recording = True  # Set stop_recording flag variable to True when you want to stop the recording
        self.ui.btnStartLive.setEnabled(True)
        self.ui.btnStartLive.setStyleSheet("background-color: rgb(12, 149, 53);")
        self.ui.btnStartLive.setStyleSheet("color: white;")
        self.ui.btnStopLive.setEnabled(False)
        self.ui.btnStopLive.setStyleSheet("background-color: #834545;")


class Transcription(QThread):
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.stop_recording = False


    
    def convert_to_mono_wav(self,filename, output_dir):
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension not in ['.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg']:
            raise ValueError("Unsupported file format")

        sound = AudioSegment.from_file(filename, format=file_extension[1:])
        sound = sound.set_channels(1)
        new_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0] + ".wav")
        sound.export(new_filename, format="wav")
        sampling_rate, wav = wavfile.read(new_filename)

        return new_filename, sampling_rate, wav
    
    def format_time(self, seconds):
        """
        Convert seconds to HH:MM:SS format
        """
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


    def run(self):

        selected_model = self.ui.selectModel.currentText()

        if selected_model == "English 1":
            trans_model = "trained-model"
        elif selected_model == "Filipino":
            trans_model = "vosk-model-tl-ph-generic-0.6"

        filename = self.ui.uploadFilename.text()
        output_dir = 'uploads'
        os.makedirs(output_dir, exist_ok=True)

        filename, sampling_rate, wav = self.convert_to_mono_wav(filename, output_dir)

        print(filename)
        print(trans_model)

        self.ui.consoleLog.append("> Transcription is Starting ")
        self.ui.consoleLog.append("> Model Selected:  " + trans_model)

        model = Model("model/" + trans_model)
        wf = wave.open(filename, "rb")

        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        # Transcribe the audio
        transcription = []
        total_frames = wf.getnframes()
        frame_rate = wf.getframerate()
        chunk_size = frame_rate  # Read in chunks of 1 second
        frames_read = 0

        while frames_read < total_frames:
            remaining_frames = total_frames - frames_read
            frames_to_read = min(chunk_size, remaining_frames)
            data = wf.readframes(frames_to_read)
            frames_read += frames_to_read
            progress = int(frames_read / total_frames * 100)
            print(f"Transcribing... {progress}% complete", end="\r")
            self.ui.consoleLog.append(f"Transcribing... {progress}% complete")
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                # Convert json output to dict
                result_dict = json.loads(rec.Result())
                # Check if the "result" key is present in the dict
                if "result" in result_dict:
                    # Extract start and end timestamps and text values
                    start = result_dict["result"][0]["start"]
                    end = result_dict["result"][-1]["end"]
                    text = " ".join([word["word"] for word in result_dict["result"]])
                    # Append dictionary to transcription list
                    transcription.append({"start": start, "end": end, "text": text})
                else:
                    # Handle the case when the "result" key is not present in the dict
                    print("No result found in transcription")

        # Get the final result after the loop
        result_dict = json.loads(rec.FinalResult())
        if "result" in result_dict:
            start = result_dict["result"][0]["start"]
            end = result_dict["result"][-1]["end"]
            text = " ".join([word["word"] for word in result_dict["result"]])
            transcription.append({"start": start, "end": end, "text": text})
        else:
            print("No result found in the final transcription")

        # Sort transcription list by start time
        transcription = sorted(transcription, key=lambda x: x["start"])

        # Output the results in the format (Timestamp) : Transcription
        for t in transcription:
            start_time = str(timedelta(seconds=int(t["start"]))).split(".")[0]
            end_time = str(timedelta(seconds=int(t["end"]))).split(".")[0]
            trans = ("%s:\n%s\n"% (f"({start_time} - {end_time})",t["text"]))
            print(trans)
            self.ui.outputText.append(trans)

            timeD.sleep(1)

                
              
    

        

class UIS_RNN(MainWindow):
    def __init__(self):
        super().__init__()
        self.thread = None

  

    def transcribeLiveStart(self):
        self.thread = SpeechLive(self.ui)
        self.thread.start()

    def stopLiveTranscription(self):
        if self.thread is not None and isinstance(self.thread, SpeechLive):
            self.thread.stopRecording()

    def transcribeOnly(self):
        self.thread = Transcription(self.ui)
        self.thread.start()
            
    def transcribeAudio(self):
            self.thread = SpeechThread(self.ui)
            self.thread.start()

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        super().closeEvent(event)

