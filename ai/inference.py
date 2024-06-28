import io

import torch
import torchaudio

from ai.cnn import CNNNetwork

SAMPLE_RATE = 8000

class_mapping = [
    "answering_machine",
    "human",
    "rings"
]

# Load trained model
cnn = CNNNetwork()
state_dict = torch.load(
    "ai/saved/voiptime.pth",
    map_location=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
)
cnn.load_state_dict(state_dict)
cnn.eval()


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


def predict_frames(audio_buffer):
    # frames.close()
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_buffer), format="s16")
    # Apply the same mel spectrogram transformation used during dataset preprocessing
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    input = mel_spectrogram(waveform)

    # Ensure the input has the expected dimensions [batch size, num_channels, frequency, time]
    input = input.unsqueeze_(0)  # Add batch and channel dimensions

    # Perform inference
    with torch.no_grad():
        output = cnn(input)

    # You may need to map the output to a class label based on your class mapping
    softmax_probs = torch.softmax(output, dim=1)

    # Iterate through each prediction (assuming 'output' has shape [batch_size, num_classes])
    for i in range(softmax_probs.size(0)):
        class_probs = softmax_probs[i]  # Get the predicted class probabilities for one input
        predicted_class = torch.argmax(class_probs).item()  # Get the predicted class
        predicted_probability = class_probs[predicted_class].item()  # Get the probability for the predicted class

        print(f"Prediction for input {i + 1} - Class: {predicted_class}, Probability: {predicted_probability:.4f}")


    predicted_class = torch.argmax(output, dim=1).item()

    # Print the predicted class
    print("Predicted class:", predicted_class)
    return class_mapping[predicted_class]


if __name__ == "__main__":
    # Define audio file path
    file_path = '/home/vkhomyn/Downloads/30.wav'

    # Load the audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Apply the same mel spectrogram transformation used during dataset preprocessing
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    input = mel_spectrogram(waveform)

    # Ensure the input has the expected dimensions [batch size, num_channels, frequency, time]
    input = input.unsqueeze_(0)  # Add batch and channel dimensions

    # Perform inference
    with torch.no_grad():
        output = cnn(input)

    # You may need to map the output to a class label based on your class mapping
    predicted_class = torch.argmax(output, dim=1).item()

    # Print the predicted class
    print("Predicted class:", predicted_class)