# audio_command_detector.py
import os
import numpy as np
import webrtcvad
import noisereduce as nr
from scipy.io import wavfile
from pydub import AudioSegment
import librosa
import torch
import torch.nn as nn
import tensorflow as tf
from tqdm import tqdm
import sounddevice as sd
from pynput import keyboard
import time


class AudioCommandDetector:
    def __init__(self, model_path="data/models/audio_classifier_best.pth"):
        # Khởi tạo model
        self.model = self._init_model()
        self.model.load_state_dict(torch.load(model_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Định nghĩa classes
        self.classes = [
            "bat_den",
            "bat_dieu_hoa",
            "bat_quat",
            "bat_tv",
            "do_am",
            "dong_rem",
            "mo_rem",
            "nhiet_do",
            "tat_den",
            "tat_dieu_hoa",
            "tat_quat",
            "tat_tv",
            "unknown",
        ]

        # Khởi tạo VAD
        self.vad = webrtcvad.Vad(2)

    def _init_model(self):
        """Khởi tạo model ConvMixer"""

        class Residual(nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x):
                return self.fn(x) + x

        def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=13):
            return nn.Sequential(
                nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
                nn.GELU(),
                nn.BatchNorm2d(dim),
                *[
                    nn.Sequential(
                        Residual(
                            nn.Sequential(
                                nn.Conv2d(
                                    dim,
                                    dim,
                                    kernel_size=kernel_size,
                                    groups=dim,
                                    padding="same",
                                ),
                                nn.GELU(),
                                nn.BatchNorm2d(dim),
                            )
                        ),
                        nn.Conv2d(dim, dim, kernel_size=1),
                        nn.GELU(),
                        nn.BatchNorm2d(dim),
                    )
                    for i in range(depth)
                ],
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(dim, n_classes),
            )

        return ConvMixer(dim=256, depth=8, n_classes=13)

    def preprocess_audio(self, input_path, output_path=None):
        """Tiền xử lý âm thanh"""
        # Đọc và chuyển sang mono, 16kHz
        audio = AudioSegment.from_wav(input_path).set_channels(1).set_frame_rate(16000)
        raw_audio = np.array(audio.get_array_of_samples())
        rate = audio.frame_rate

        # Lọc nhiễu
        denoised_audio = nr.reduce_noise(y=raw_audio.astype(np.float32), sr=rate)

        # Resample nếu cần
        if rate != 16000:
            raw_audio = librosa.resample(
                raw_audio.astype(np.float32), orig_sr=rate, target_sr=16000
            )
            denoised_audio = librosa.resample(
                denoised_audio.astype(np.float32), orig_sr=rate, target_sr=16000
            )
            rate = 16000

        # VAD
        frame_duration_ms = 30
        frame_length = int(rate * frame_duration_ms / 1000)
        frames = [
            denoised_audio[i : i + frame_length]
            for i in range(0, len(denoised_audio) - frame_length, frame_length)
        ]

        def is_speech(frame):
            int16_frame = (frame * 32768).astype(np.int16)
            return self.vad.is_speech(int16_frame.tobytes(), rate)

        flags = [is_speech(frame) for frame in frames]
        speech_mask = np.repeat(flags, frame_length)
        speech_mask = np.pad(
            speech_mask, (0, len(denoised_audio) - len(speech_mask)), mode="constant"
        )
        speech_audio = denoised_audio * speech_mask

        # CAD: 1.5s đoạn có năng lượng cao
        window_sec = 1.5
        window_len = int(window_sec * rate)
        stride = int(0.2 * rate)

        max_energy = 0
        best_segment = None
        for i in range(0, len(speech_audio) - window_len, stride):
            window = speech_audio[i : i + window_len]
            energy = np.sum(window.astype(np.float32) ** 2)
            if energy > max_energy:
                max_energy = energy
                best_segment = window

        final_segment = best_segment

        segment_len = int(1.0 * rate)
        stride = int(0.02 * rate)

        max_energy = 0
        best_start = 0
        for i in range(0, len(final_segment) - segment_len + 1, stride):
            window = final_segment[i : i + segment_len]
            energy = np.sum(window.astype(np.float32) ** 2)
            if energy > max_energy:
                max_energy = energy
                best_start = i

        padded_segment = final_segment[best_start : best_start + segment_len]

        # Chuẩn hóa âm lượng (Peak Normalization)
        max_val = np.max(np.abs(padded_segment))
        if max_val > 0:
            padded_segment = padded_segment / max_val * 0.99

        # Chuyển về int16 để ghi WAV
        final_output = (padded_segment * 32767).astype(np.int16)

        # Ghi file WAV nếu có output_path
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            wavfile.write(output_path, rate, final_output)

        return final_output, rate

    def extract_mel_spectrogram(
        self, audio_data, sr=16000, n_mels=128, n_fft=2048, hop_length=128
    ):
        """Trích xuất mel spectrogram"""
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data.astype(np.float32),
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=20,
            fmax=sr / 2,
            power=2.0,
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Chuẩn hóa về khoảng [0,1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (
            mel_spec_db.max() - mel_spec_db.min()
        )

        # Resize về kích thước cố định (128, 32)
        mel_spec_norm = tf.image.resize(mel_spec_norm[..., np.newaxis], (128, 32))
        mel_spec_norm = mel_spec_norm.numpy()
        mel_spec_norm = mel_spec_norm[..., 0]

        return mel_spec_norm

    def predict(self, audio_path):
        """Dự đoán lệnh từ file âm thanh"""
        # Tiền xử lý âm thanh
        audio_data, sr = self.preprocess_audio(audio_path)

        # Trích xuất mel spectrogram
        mel_features = self.extract_mel_spectrogram(audio_data, sr)

        # Chuẩn bị input cho model
        features = (
            torch.FloatTensor(mel_features).unsqueeze(0).unsqueeze(0).to(self.device)
        )

        # Dự đoán
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Lấy top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities[0], 3)
        top3_predictions = [
            (self.classes[idx], prob.item())
            for idx, prob in zip(top3_indices, top3_prob)
        ]

        return {
            "predicted_class": self.classes[predicted_class],
            "confidence": confidence,
            "top3_predictions": top3_predictions,
        }


def record_audio(duration=3, sample_rate=16000):
    """Ghi âm với độ dài cố định và đợi nút play"""
    print("Nhấn Enter để bắt đầu ghi âm...")
    input()  # Đợi người dùng nhấn Enter

    print("Đang ghi âm...")

    try:
        # Ghi âm
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()

        # Lưu file
        output_path = "command.wav"
        wavfile.write(output_path, sample_rate, recording)
        print(f"Đã lưu file âm thanh tại: {output_path}")
        return output_path
    except Exception as e:
        print(f"Lỗi khi ghi âm: {str(e)}")
        return None


def main():
    # Khởi tạo detector
    detector = AudioCommandDetector()
    while True:
        try:
            # Ghi âm khi nhấn Enter
            audio_file = record_audio()

            if audio_file is None:
                print("Không thể ghi âm. Thử lại...")
                continue

            # Dự đoán lệnh
            result = detector.predict(audio_file)

            print("\nKết quả dự đoán:")
            print(f"Lệnh: {result['predicted_class']}")
            print(f"Độ tin cậy: {result['confidence']:.2%}")
            print("\nTop 3 dự đoán:")
            for class_name, prob in result["top3_predictions"]:
                print(f"{class_name}: {prob:.2%}")

            print("\nNhấn Ctrl+C để thoát hoặc Enter để ghi âm tiếp...")

        except KeyboardInterrupt:
            print("\nĐã thoát chương trình.")
            break
        except Exception as e:
            print(f"Lỗi: {str(e)}")
            continue


if __name__ == "__main__":
    main()
