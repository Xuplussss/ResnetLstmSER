import torch
import torch.nn as nn
import os, datetime, math, tgt, random, time
import numpy as np
from sklearn.metrics import confusion_matrix
from lib.LSTM_attention import Seq2Seq_attention_BLSTM
from lib.audio import trim_audio
from nltk.translate import bleu_score as bleu
random.seed(2708)
torch.manual_seed(2708)
np.random.seed(2708)
begin_time = datetime.datetime.now()
########################### note ############################
# EMOTION = {0:'anger', 1:'anxiety', 2:'sadness', 3:'surprise', 4:'neutral', 5:'boredom', 6:'happiness', 7:'silence'}
# SOUND = {0:'laugh', 1:'breath', 2:'shout', 3:'silence', 4:'verbal'}
#############################################################

# Enviroment Parameters
date = time.localtime(time.time())
SAVE_DATE = str(date.tm_year)+'_'+str(date.tm_mon)+'_'+str(date.tm_day)+'_'
SAVE_FOLD = True
USE_CUDA = True
TG_PATH = 'data/tag/'

WAV_PATH = 'data/audio/'

MODEL_PATH = './model/'
RESULT_PATH = './result/LSTM/'


sound_model = '.pt'
emotion_model = '.pt'

RESULT_FILE = '_seq2seq_result.txt'
EMOTION_TYPE = {0:'anger', 1:'anxiety', 2:'sadness', 3:'surprise', 4:'neutral', 5:'boredom', 6:'happiness', 7:'silence'}
SOUND_TYPE = {0:'laugh', 1:'breath', 2:'shout', 3:'silence', 4:'verbal'}

# Hyper Parameters
FOLD, EPOCH, BATCH_SIZE = 5, 30, 100
SND_CNN_FILTER, SND_CNN_POOL = 50, 2
EMO_CNN_FILTER, EMO_CNN_POOL = 300, 10
LR = 0.01
INPUT_DIM = SND_CNN_POOL*SND_CNN_FILTER + EMO_CNN_POOL*EMO_CNN_FILTER
HIDDEN_DIM = [512, 256, 128, 64, 32]
HIDDEN_DIM = [32, 64, 128, 256, 512]

# Best Result Record
BEST_ACC = 0
BEST_FILE = ''
BEST_LOSS = 5
BEST_LFILE = ''

def all_neutral_silence(tier):
	silence_dur = []
	neutral_dur = []
	all_ne_si = True
	for i in range(len(tier)):
		if tier[i].text[0] in ['0','1','2']:
			all_ne_si = False
			break
		elif tier[i].text[1] not in ['4','7']:
			all_ne_si = False
			break
		elif tier[i].text[1] == '4':
			neutral_dur.append(tier[i].end_time-tier[i].start_time)
		elif tier[i].text[1] == '7':
			silence_dur.append(tier[i].end_time-tier[i].start_time)
	return all_ne_si, neutral_dur, silence_dur

def read_input(wavpath, tgpath):
	wavs = []
	tags = []
	sound_tags = []
	type_dis = [0]*len(EMOTION_TYPE)
	sound_dis = [0]*len(SOUND_TYPE)
	all_ne_si_count = 0
	ne_si_wavs = []
	ne_si_tags = []
	for file in sorted(os.listdir(tgpath)):
		if file.endswith('.TextGrid'):
			tg = tgt.read_textgrid(tgpath+file)
			tag_tier = tg.get_tier_by_name('silences')
			all_ne_si, ne_dur, si_dur = all_neutral_silence(tag_tier)
			boundary = [i.start_time for i in tag_tier]
			boundary.append(tag_tier[-1].end_time)
			sound_tags.append([int(i.text[0]) for i in tag_tier])
			tag = np.array([[int(i.text[1])] for i in tag_tier])
			pph_audio = trim_audio(wavpath+file[:-9]+'.wav', boundary)
			if all_ne_si:
				all_ne_si_count += 1
				if sum(ne_dur)-sum(si_dur) < 0:
					ne_si_wavs.append(pph_audio)
					ne_si_tags.append(tag)
					continue
				
				else:
					if random.random() >= 0.4:
						ne_si_wavs.append(pph_audio)
						ne_si_tags.append(tag)
						continue
			for item in sound_tags[-1]:
				sound_dis[item] += 1
			for i in tag:
				type_dis[i[0]] += 1
			tags.append(tag)
			wavs.append(pph_audio)
	print('num of all_si_ne:', all_ne_si_count)
	print('number of pph:', sum(type_dis), ' dis:', type_dis)
	print('sound_dis:', sound_dis)
	print('original number of turns:',len(os.listdir(tgpath)))
	print('number of turns:',len(wavs))
	wavs = np.array(wavs)
	tags = np.array(tags)
	sound_tags = np.array(sound_tags)
	return wavs, tags, sound_tags, sound_dis, type_dis, ne_si_wavs, ne_si_tags

def write_result(result_path, result_file, fold_acc, fold_loss, fold_bleu, fold_conf, epoch, batch, lr, tag, model, run_time):
	wf = open(result_path+result_file, 'w')
	wf.write('model:\n')
	wf.write(str(model))
	wf.write('\n')
	wf.write('run time:'+str(run_time)+'\n')
	wf.write('epoch:   '+str(epoch)+'\n')
	wf.write('batch:   '+str(batch)+'\n')
	wf.write('learn rate:'+str(lr)+'\n')
	wf.write('tag type:'+str(tag)+'\n')
	for i in range(len(fold_acc)):
		wf.write(str(fold_conf[i])+'\n')
		wf.write('accuracy: '+str(fold_acc[i])+'| loss: '+str(fold_loss[i])+'\n')
		wf.write('mean of bleu: '+str(fold_bleu[i])+'\n')
	wf.write('mean of accuracy:'+str(sum(fold_acc)/5)+' | loss: '+str(sum(fold_loss)/5)+'\n')
	wf.close()

def main():
	wavs, tags, sound_tags, snd_dis, tag_dis, ne_si_wavs, ne_si_tags = read_input(WAV_PATH, TG_PATH)
	device = torch.device('cuda:0' if USE_CUDA else 'cpu')
	# device = torch.device('cpu')
	cnn_sound = torch.load(MODEL_PATH+sound_model).to(device)
	cnn_emotion = torch.load(MODEL_PATH+emotion_model).to(device)
	cnn_sound.eval()
	cnn_emotion.eval()
	all_result = []
	cnn_start = datetime.datetime.now()
	print('get cnn feature...')
	seqs = []
	for i in range(len(wavs)):
		seq_hidden = []
		for j in range(len(wavs[i])):
			wav_tensor = torch.from_numpy(wavs[i][j][np.newaxis,np.newaxis,:]).to(device)
			hidden_emo, output_emo = cnn_emotion(wav_tensor)
			hidden_snd, output_snd = cnn_sound(wav_tensor)
			hidden = torch.cat((hidden_snd, hidden_emo),1)
			hidden.detach_().to(device)
			seq_hidden.append(hidden)
		# input of shape (seq_len, batch, input_size)
		lstm_input = torch.cat(seq_hidden).view(len(seq_hidden),1,-1).to(device).float()
		seqs.append(lstm_input)		
	print('cnn cost:',datetime.datetime.now()-cnn_start)
	smooth = bleu.SmoothingFunction()

	ne_si_seqs = []
	for i in range(len(ne_si_wavs)):
		seq_hidden = []
		for j in range(len(ne_si_wavs[i])):
			wav_tensor = torch.from_numpy(ne_si_wavs[i][j][np.newaxis, np.newaxis, :]).to(device)
			hidden_emo, output_emo = cnn_emotion(wav_tensor)
			hidden_snd, output_snd = cnn_sound(wav_tensor)
			hidden = torch.cat((hidden_snd, hidden_emo),1)
			hidden.detach_().to(device)
			seq_hidden.append(hidden)
		lstm_input = torch.cat(seq_hidden).view(len(seq_hidden),1,-1).to(device).float()
		ne_si_seqs.append(lstm_input)

	for hidden_i in range(len(HIDDEN_DIM)):
		fold_acc, fold_loss, fold_bleu = [0]*FOLD, [0]*FOLD, [0]*FOLD
		fold_conf = []

		for fold in range(FOLD):
			train_start = datetime.datetime.now()
			lstm = Seq2Seq_attention_BLSTM(input_dim = INPUT_DIM, hidden_dim = HIDDEN_DIM[hidden_i], output_dim = len(EMOTION_TYPE.keys()))
			lstm.to(device).float()
			optimizer = torch.optim.Adam(lstm.parameters(), lr = LR)
			# loss_func = nn.NLLLoss()
			loss_func = nn.CrossEntropyLoss()
			train_seq, train_tag, train_type = [], [], []
			test_seq, test_tag, test_type = [], [], []
			for item in range(len(seqs)):
				if item % 5 == fold:
					test_seq.append(seqs[item])
					test_tag.append(tags[item])
				else:
					train_seq.append(seqs[item])
					train_tag.append(tags[item])
			
			for epoch in range(EPOCH):
				result = []
				batchi = 0
				lastloss = 0

				for i in range(len(train_seq)):
					de_out, de_h_n, weight = lstm(train_seq[i], device)
					# h_n of lstm(num_layers * num_directions, batch, hidden_size) = (1,1,8)
					result.append(de_out)
		
					if (len(result)==BATCH_SIZE)|(i+1==len(train_seq)):
						losses = []
						for index in range(i+1-batchi*BATCH_SIZE):
							tag_va = torch.from_numpy(train_tag[batchi*BATCH_SIZE+index]).view(-1).to(device)
							
							loss = loss_func(result[index].view(result[index].size(0),-1), tag_va)
							losses.append(loss.view(1))
						batchi += 1
						result = []
						optimizer.zero_grad()
						loss_batch = torch.cat(losses).mean().to(device)
						loss_batch.backward()
						optimizer.step()
						lastloss += loss_batch.item()
				lastloss = lastloss/math.ceil(len(train_seq)/BATCH_SIZE)
				
				acc_count = 0
				result_pre = []
				conf_tag = []
				for i in range(len(test_seq)):
					de_out, de_h_n, weight = lstm(test_seq[i], device)
					pred_tag = torch.max(de_out.view(de_out.size(0),-1),1)[1].to(device).view(de_out.size(0),-1)
					for ind, v in enumerate(pred_tag):
						result_pre.append(v.item())
						conf_tag.append(test_tag[i][ind])
						if v.item()==test_tag[i][ind]:
							acc_count += 1
				for i in range(len(ne_si_seqs)):
					de_out, de_h_n, weight = lstm(ne_si_seqs[i], device)
					pred_tag = torch.max(de_out.view(de_out.size(0),-1),1)[1].to(device).view(de_out.size(0),-1)
					for ind, v in enumerate(pred_tag):
						result_pre.append(v.item())
						conf_tag.append(ne_si_tags[i][ind])
						if v.item()==ne_si_tags[i][ind]:
							acc_count += 1

				acc = acc_count/(len(conf_tag))
				confusion = confusion_matrix(conf_tag,result_pre)
				# print('ffmpeg 10db')
				print('whole training train with bleu and ne_si as test')
				print('sound',snd_dis)
				print('emotion',tag_dis)
				print(confusion)
				print('whole:Hidden:',HIDDEN_DIM[hidden_i],'Fold:',fold,' Epoch:', epoch,'| train loss: %.6f' % lastloss, '| test accuracy: %.4f ' % acc)
			
			bleus = []
			for i in range(len(test_seq)):
				de_out, de_h_n, weight = lstm(test_seq[i], device)
				pred_tag = torch.max(de_out.view(de_out.size(0),-1),1)[1].to(device).view(de_out.size(0),-1)
				bleu_hyp = [EMOTION_TYPE[tag.item()] for tag in pred_tag]
				bleu_ref = [EMOTION_TYPE[tag[0]] for tag in test_tag[i]]
				# bleus.append(100*bleu.sentence_bleu([bleu_ref], bleu_hyp, smoothing_function = None, weights = (0.4, 0.4, 0.2, 0)))
				bleus.append(100*bleu.sentence_bleu([bleu_ref], bleu_hyp, smoothing_function = smooth.method2, weights = (0.4, 0.4, 0.2, 0)))
			print('mean of bleu:',sum(bleus)/len(bleus))
			if SAVE_FOLD:
				model_fold = SAVE_DATE+'_whole_7emoSIL_SNDp'+str(SND_CNN_POOL)+'f'+str(SND_CNN_FILTER)+'_EMOp'+str(EMO_CNN_POOL)+'f'+str(EMO_CNN_FILTER)+'_hidden'+str(HIDDEN_DIM[hidden_i])+'_fold'+str(fold)+'.pt'
				torch.save(lstm,MODEL_PATH+model_fold)
			fold_acc[fold] = acc
			fold_loss[fold] = lastloss
			fold_bleu[fold] = sum(bleus)/len(bleus)
			fold_conf.append(confusion)
			train_end = datetime.datetime.now()
			print(lstm)
			train_time = train_end-train_start
			print('training time:',train_time)
		txt_name = SAVE_DATE+'_whole_7emoSIL_SNDp'+str(SND_CNN_POOL)+'f'+str(SND_CNN_FILTER)+'_EMOp'+str(EMO_CNN_POOL)+'f'+str(EMO_CNN_FILTER)+'_hidden'+str(HIDDEN_DIM[hidden_i])+RESULT_FILE
		write_result(RESULT_PATH, txt_name, fold_acc, fold_loss, fold_bleu, fold_conf, EPOCH, BATCH_SIZE, LR, EMOTION_TYPE, lstm, train_time)
		all_result.append([fold_acc, fold_loss, fold_conf, lstm, train_time, HIDDEN_DIM[hidden_i]])
		if np.mean(fold_acc) > BEST_ACC:
			BEST_ACC = np.mean(fold_acc)
			BEST_FILE = txt_name
			print(BEST_ACC,' ',BEST_FILE)
		if np.mean(fold_loss) < BEST_LOSS:
			BEST_LOSS = np.mean(fold_loss)
			BEST_LFILE = txt_name
		
	end_time = datetime.datetime.now()

	wf = open('./'+SAVE_DATE+'whole_seq2seq_best.txt','w')
	wf.write('maximum accuracy file:'+BEST_FILE)
	wf.write('\naccuracy: '+str(BEST_ACC))
	wf.write('\nminimum loss file:'+BEST_LFILE)
	wf.write('\nloss: '+str(BEST_LOSS))
	wf.write('\ntraining time:'+str(end_time-begin_time))
	wf.write('\n')
	for i in all_result:
		wf.write('*********************************\n')
		wf.write('hidden size: '+str(i[-1])+' train time:'+str(i[-2])+'\n')
		wf.write('accuracy: '+str(i[0])+'\n')
		wf.write('loss:     '+str(i[1])+'\n')
	wf.close()


if __name__ == "__main__":
    main()