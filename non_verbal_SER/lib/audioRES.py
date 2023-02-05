import scipy.io.wavfile
import librosa
import numpy as np 
import wave

def wav_len(file):
	wf = wave.open(file,'rb')
	return wf.getnframes()/float(wf.getframerate())

def trim_audio(file, boundary):
	try:
		# fstmp, wavtmp = scipy.io.wavfile.read(file)
		y, sr = librosa.load(file)
	except ValueError:
		print('read audio error:',file,' has unexpected end.')
	#wavtmp.dtype = int16
	nb_bits = 16
	# samples = wavtmp / (float(2**(nb_bits - 1)) + 1.0)
	# samples.reshape(samples.shape[0],1) 
	sample_boundary = [int(i*sr) for i in boundary]
	pph_audio = []
	for i in range(len(boundary)-1):
		pph_audio.append(np.real(librosa.stft(y[sample_boundary[i]:sample_boundary[i+1]])))
	
	return pph_audio