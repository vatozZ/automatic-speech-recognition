import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

FIG_SIZE = (15,10)

#hedef dosya
file = ""
# sesin yüklenmesi, örneklenmesi
signal, sample_rate = librosa.load(file, sr=22050)

#dalga formu, zaman domaini
plt.figure(figsize=FIG_SIZE)
librosa.display.waveplot(signal, sample_rate, alpha=0.4)
plt.xlabel("zaman(s)")
plt.ylabel("Genlik")
# fourier transform
fft = np.fft.fft(signal)
#  mutlak değerini alarak komplex ifadelerden kurtuluyoruz
spectrum = np.abs(fft)
#frekans değişkeni tanımlanması
f = np.linspace(0, sample_rate, len(spectrum))

left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]

plt.figure(figsize=FIG_SIZE)
plt.plot(left_f, left_spectrum, alpha=0.4)
plt.xlabel("Frekans")
plt.ylabel("Genlik")
plt.title("Güç Spektrumu")

hop_length = 512
n_fft = 2048

#hop_length_duration = float(hop_length)/sample_rate
#n_fft_duration = float(n_fft)/sample_rate

stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
spectrogram = np.abs(stft)

plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("zaman(s)")
plt.ylabel("frekans(f)")
plt.colorbar()
plt.title("Spektrogram")

log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("zaman(s)")
plt.ylabel("frekans(f)")
plt.colorbar(format="%+2.0f dB")
plt.title("spektrogram(dB)")

MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
plt.xlabel("zaman")
plt.ylabel("MFCC")
plt.colorbar()
plt.title("MFCCs, Mel Frekans Kepstral Katsayıları")
plt.show()



