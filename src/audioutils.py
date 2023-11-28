import pickle
import re

import librosa.display
import numpy as np
import torch
from matplotlib import pyplot as plt

from ai.models import ESC50Model


class VoicemailRecognitionAudioUtil:
    """ This class is responsible for parsing config
        and building recognition criteria

           Initiated at the entry point
    """

    def __init__(self):
        self.silence = "silence"
        self.ring = "ring"
        self.pattern = r'<(.*?)\s*\|\s*([^>]+)>'
        self.cnn = ESC50Model(input_shape=(1, 128, 431))
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.state_dict = torch.load("ai/saved/voiptime3.pth", map_location=self.device)
        self.cnn.load_state_dict(self.state_dict)
        self.cnn.eval()
        with open('ai/saved/indexer3.pkl', 'rb') as f:
            self.indexer = pickle.load(f)

    def predict_sound_from_bytes(self, file_name):
        spec = self.get_spectrogram_db(file_name)
        # self.draw_spectrogram(spec)
        spec_norm = self.spec_to_image(spec)
        spec_t = torch.tensor(spec_norm).to(self.device, dtype=torch.float32)
        pr = self.cnn.forward(spec_t.reshape(1, 1, *spec_t.shape))[0].cpu().detach().numpy()
        pred = {name: pr[ind] for ind, name in self.indexer.items()}
        res = list(pred.items())
        s = 0
        for c, val in res:
            s += np.exp(val)
        for i in range(len(res)):
            res[i] = (res[i][0], np.exp(res[i][1]) / s)
        res.sort(key=lambda x: x[1], reverse=True)
        return dict(res)

    @staticmethod
    def draw_spectrogram(spec):
        plt.figure()
        plt.subplot(3, 1, 1)
        # librosa.display.waveshow(wav, sr=sr, color="red")
        librosa.display.specshow(spec, cmap='viridis')
        plt.show()

    @staticmethod
    def spec_to_image(spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled

    @staticmethod
    def get_spectrogram_db(buffer, sr=None, n_fft=2048, hop_length=94, n_mels=128, fmin=20, fmax=4150, top_db=80):
        clip, sr = librosa.load(buffer, sr=sr)
        # Remove silimce 4 decebels
        wav = librosa.effects.trim(clip, top_db=4)[0]
        if wav.shape[0] < 5 * sr:
            wav = np.pad(wav, int(np.ceil((5 * sr - wav.shape[0]) / 2)), mode='reflect')
        else:
            wav = wav[:5 * sr]
        spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                                              fmin=fmin, fmax=fmax)
        spec_db = librosa.power_to_db(spec, top_db=top_db)
        return spec_db

    def parse_config_prediction_criteria(self, predictions, iterations):
        detected = dict()
        for match in re.findall(self.pattern, predictions):
            first_values = [value.strip() for value in match[0].split(',')]
            result = match[1].strip()
            phrase = ''
            for index, first in enumerate(first_values):
                if index < iterations:
                    phrase += first
            detected[phrase] = result
        return detected

    def build_interim_condition(self, prediction):
        transcript = str(max(prediction, key=prediction.get)) if prediction else self.silence
        confidence = prediction[transcript] if prediction else 1.0
        return transcript, confidence

    def is_ring_condition(self, prediction):
        return str(max(prediction, key=prediction.get)) == self.ring

    def build_schema_condition(self, predictions, prediction_criteria):
        key = ''
        value = 0.0
        for prediction in predictions:
            maximum = str(max(prediction, key=prediction.get))
            key += maximum
            value += prediction[maximum]
        result = prediction_criteria[key]
        if not result:
            return self.silence
        # Divide recognition results by len of predictions to count avg value
        return result, key.count(result) / len(predictions)

    def parse_config_from_request(self, app, stt_config):
        max_interval = stt_config.interim_results_config.interval
        max_predictions = stt_config.interim_results_config.max_predictions
        prediction_criteria = stt_config.interim_results_config.prediction_criteria
        sample_rate_hertz = stt_config.config.sample_rate_hertz
        configurations = {
            'encoding': stt_config.config.encoding,
            'sample_rate_hertz': sample_rate_hertz if sample_rate_hertz is not None else app.audio_sample_rate,
            'language_code': stt_config.config.language_code,
            'max_alternatives': stt_config.config.max_alternatives,
            'profanity_filter': stt_config.config.profanity_filter,
            'enable_automatic_punctuation': stt_config.config.enable_automatic_punctuation,
            'num_channels': stt_config.config.num_channels,
            'do_not_perform_vad': stt_config.config.do_not_perform_vad,
            'vad_config': {
                'min_speech_duration': stt_config.config.vad_config.min_speech_duration,
                'max_speech_duration': stt_config.config.vad_config.max_speech_duration,
                'silence_duration_threshold': stt_config.config.vad_config.silence_duration_threshold,
                'silence_prob_threshold': stt_config.config.vad_config.silence_prob_threshold,
                'aggressiveness': stt_config.config.vad_config.aggressiveness
            },
            'interim_results_config': {
                'enable_interim_results': stt_config.interim_results_config.enable_interim_results,
                'max_interval': max_interval if max_interval is not None else app.audio_interval,
                'max_predictions': max_predictions if max_predictions is not None else app.max_predictions,
                'prediction_criteria': prediction_criteria if prediction_criteria is not None else app.prediction_criteria,
            },
            'enable_sentiment_analysis': stt_config.config.enable_sentiment_analysis,
            'enable_gender_identification': stt_config.config.enable_gender_identification,
            'extension': stt_config.config.channel_exten
        }

        configurations['prediction_criteria'] = self.parse_config_prediction_criteria(
            configurations['interim_results_config']['prediction_criteria'],
            configurations['interim_results_config']['max_predictions']
        )
        configurations['desired_num_samples'] = (
                configurations['sample_rate_hertz'] * configurations['interim_results_config']['max_interval']
        )
        return configurations

    @staticmethod
    def swap_zero_bytes(silence_chunks, chunks):
        return chunks.replace(silence_chunks, b'')
