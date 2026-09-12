import torch

__asr_model = None
__stt_model = None


def calculate_word_error_rate(audio1, audio2):
    import whisper
    from jiwer import wer

    global __stt_model

    # Load Whisper
    if not __stt_model:
        __stt_model = whisper.load_model("large-v3", device=torch.device("cuda"))

    # Transcribe both audios
    result1 = __stt_model.transcribe(audio1["array"])
    result2 = __stt_model.transcribe(audio2["array"])

    reference = result1["text"]
    hypothesis = result2["text"]

    # Calculate WER
    error_rate = wer(reference, hypothesis)

    return error_rate

def calculate_speaker_similarity(audio1, audio2):
    # instantiate pretrained model
    from pyannote.audio import Model

    global __asr_model

    if not __asr_model:
        __asr_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    # Better Alternative: "pyannote/embedding", use_auth_token=os.env["HF_TOKEN"]

    from pyannote.audio import Inference
    inference = Inference(__asr_model, window="whole")
    inference.to(torch.device("cuda"))

    embedding1 = inference(__to_waveform(audio1))
    embedding2 = inference(__to_waveform(audio2))
    # `embeddingX` is (1 x D) numpy array extracted from the file as a whole.

    from scipy.spatial.distance import cosine
    distance = cosine(embedding1, embedding2)
    # `distance` is a `float` describing how dissimilar speakers 1 and 2 are.
    return 1-distance

def __to_waveform(audio):
    waveform = torch.from_numpy(audio["array"]).float()

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    return {
        "waveform": waveform,
        "sample_rate": audio["sampling_rate"],
    }
