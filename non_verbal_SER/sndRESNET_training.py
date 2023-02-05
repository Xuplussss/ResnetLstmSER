import torch
import torch.nn as nn
import os, datetime, math, tgt, random, time
from resnet import ResNet18
import numpy as np
from sklearn.metrics import confusion_matrix
from lib.audioRES import trim_audio
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
SAVE_FOLD = False
SAVE_BESTMODEL = True
USE_CUDA = True
TG_PATH = 'data/tag/'

WAV_PATH = 'data/audio/'
RESULT_PATH = './result/sound_type_RESNET/'
MODEL_PATH = './model/'
RESULT_FILE = '_RESsnd.txt'
DATA_TYPE = {0:'laugh', 1:'breath', 2:'shout', 3:'silence', 4:'verbal'}

# Hyper Parameters
FOLD, EPOCH, BATCH_SIZE = 5, 40, 4
LR, DROPOUT = 0.01, 0.3 # 0.032/0.016 sec as a frame/shift
COV_KERNEL_SIZE = [8]
COV_STEP_SIZE = [int(i/2) for i in COV_KERNEL_SIZE]
FILTER_NUM = [25]
ADAPTIVE_SIZE = [1]

# Best Result Record
BEST_ACC = 0
BEST_FILE = ''
BEST_LOSS = 5
BEST_LFILE = ''

def read_input(tg_path, wav_path, data_type):
	wavs = []
	tags = []
	time_dis = [0]*len(data_type.keys())
	type_dis = [0]*len(data_type.keys())
	for file in os.listdir(wav_path):
		if file.endswith('.wav'):
			tg = tgt.read_textgrid(tg_path+file[:-3]+'TextGrid')
			tag_tier = tg.get_tier_by_name('silences')
			boundary = [i.start_time for i in tag_tier]
			boundary.append(tag_tier.end_time)
			try:
				seg_tag = [int(i.text[1]) for i in tag_tier]
				sound_tag = [int(i.text[0]) for i in tag_tier]
			except:
				print(file)
			pph_audio = trim_audio(wav_path+file, boundary)
			for i in range(len(seg_tag)):
				# if len(pph_audio[i]) > 0.5 * 16000:
				if sound_tag[i] == 3:
					if random.random() <= 0.1:
						time_dis[3] += boundary[i+1] - boundary[i]
						wavs.append(pph_audio[i])
						tags.append([3])
						type_dis[3] += 1
				elif sound_tag[i] in data_type.keys():
					time_dis[sound_tag[i]] += boundary[i+1] - boundary[i]
					wavs.append(pph_audio[i])
					tags.append([sound_tag[i]])
					type_dis[sound_tag[i]] += 1
	print('type_dis:',type_dis)
	print('time dis:',time_dis)
	# wavs = np.array(wavs)
	# tags = np.array(tags)
	return wavs, tags, type_dis

def order_input(data, tag, type_number):
	ordered_wav = []
	ordered_tag = []
	datatype_dis = [0]*type_number
	for i in range(type_number):
		for j in range(len(data)):
			if(tag[j][0]==i):
				datatype_dis[i]+=1
				ordered_wav.append(data[j])
				ordered_tag.append(tag[j])
	print('Distrubution of types:',datatype_dis)
	# Distrubution of old emotion types: [203, 134, 177, 173, 197, 1144, 145, 165]
	return ordered_wav, ordered_tag

def random_index(data_array, tag_array):
	index = np.arange(len(data_array))
	np.random.shuffle(index)
	data_array = data_array[index]
	tag_array = tag_array[index]
	print("random file num:",len(data_array))
	return data_array, tag_array

def train_test_data(ordered_wav, ordered_tag, fold_num, fold_index):
	test_wav, test_tag, train_wav, train_tag = [], [] ,[], []
# neutral count
	neutral_count = 0
	for item in range(len(ordered_wav)):
		if(item%fold_num==fold_index):
			test_wav.append(ordered_wav[item])
			test_tag.append(ordered_tag[item])
		else:
			train_wav.append(ordered_wav[item])
			train_tag.append(ordered_tag[item])
			if ordered_tag[item][0] in [2] and random.random() <= 0.5:
				train_wav.append(ordered_wav[item])
				train_tag.append(ordered_tag[item])
				
	return train_wav, train_tag, test_wav, test_tag

def train_model(train_wav, train_tag, batch_size, model, optimizer, loss_func, device):
	result = []
	batchi = 0
	lastloss = 0
	# train_wav, train_tag = random_index(train_wav, train_tag)

	for i in range(len(train_wav)):
		wav_tensor = torch.from_numpy(train_wav[i][np.newaxis,np.newaxis,]).to(device)
		output, last = model(wav_tensor)
		result.append(output)
		
		if (len(result)==batch_size)|(i+1==len(train_wav)):
			result_va = torch.cat(result,0)
			train_tag = np.asarray(train_tag)
			tag_va = torch.from_numpy(train_tag[batchi*batch_size:i+1]).view(i+1-batchi*batch_size).to(device)
			batchi+=1
			result=[]
			optimizer.zero_grad()
			loss = loss_func(result_va,tag_va)
			loss.backward()
			optimizer.step()
			lastloss += loss.item()
	lastloss = lastloss/math.ceil(len(train_wav)/batch_size)
	return model, lastloss

def write_result(result_path, result_file, fold_acc, fold_loss, fold_conf, epoch, batch, lr, tag, model, run_time):
	# parameter, accuracy, loss, epoch, batch, trainingtime, lr, dropout, bestparameter
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
	wf.write('mean of accuracy:'+str(sum(fold_acc)/5)+' | loss: '+str(sum(fold_loss)/5)+'\n')
	wf.close()

def main():
	wavs = []
	tags = []
	ordered_wav = []
	ordered_tag = []
	########################read input############################
	# read labels with format: 'filename label' or 'filename label\n'
	# filename cannot end with type of file, e.g. '.wav'
	wavs, tags, tag_dis = read_input(TG_PATH, WAV_PATH, DATA_TYPE)
	print('total data:',len(wavs))

	device = torch.device("cuda:0" if USE_CUDA else "cpu")
	fold_acc = np.zeros(FOLD)
	fold_loss = np.zeros(FOLD)
	fold_confusion = []
	all_result = []
	# [fold_acc, fold_loss, fold_conf, model, runtime]
	for COV_i in range(len(COV_KERNEL_SIZE)):
		for filter_i in range(len(FILTER_NUM)):
			train_start = datetime.datetime.now()
			for fold in range(FOLD):
				
				res = ResNet18()
				# cnn.double().to(device)
				res.float().to(device)			
				optimizer = torch.optim.Adam(res.parameters(),lr = LR)

				train_wav, train_tag, test_wav, test_tag = train_test_data(wavs, tags, FOLD, fold)
				test_tag = np.array(test_tag)
				train_tag = np.array(train_tag)
				train_dis = [0]*len(DATA_TYPE.keys())
				train_dis_weight = [0]*len(DATA_TYPE.keys())
				for i in range(len(train_wav)):
					train_dis[train_tag[i][0]] += 1
				test_dis = [0]*len(DATA_TYPE.keys())
				for i in range(len(test_wav)):
					test_dis[test_tag[i][0]] += 1	
				
				for i in range(len(train_dis)):
					train_dis_weight[i] = 1/train_dis[i]
				
				loss_func = nn.CrossEntropyLoss()

				for epoch in range(EPOCH):
					res, lastloss = train_model(train_wav, train_tag, BATCH_SIZE, res, optimizer, loss_func, device)
					
					acc_count = 0
					result_dis = [0]*len(DATA_TYPE)
					result_pre = []
					res.eval()
					for i in range(len(test_wav)):
						test_wav_tensor = torch.from_numpy(test_wav[i][np.newaxis,np.newaxis,:]).to(device)
						test_out, testlast = res(test_wav_tensor)
						pred_tag = torch.max(test_out,1)[1].to(device).squeeze()
						result_dis[pred_tag.item()]+=1
						result_pre.append(pred_tag.item())
						if pred_tag.item()==test_tag[i]:
							acc_count +=1
					res.train()
					acc = acc_count/(len(test_wav))
					confusion = confusion_matrix(test_tag,result_pre)
					print([DATA_TYPE[i] for i in range(len(DATA_TYPE))])
					print('emotionres trained with ffmpeg 10dB')
					print('type:',tag_dis)
					print('train_dis:',train_dis)
					print('train class weight:',train_dis_weight)
					print('test_dis:',test_dis)
					print('kernel:',COV_KERNEL_SIZE[COV_i],' filter',FILTER_NUM[filter_i])
					print('Fold:   ',fold,'| Epoch:   ', epoch,'| train loss: %.6f' % lastloss, '| test accuracy: %.4f ' % acc)
					print(confusion)
				
				fold_confusion.append(confusion)
				fold_acc[fold] = acc
				fold_loss[fold] = lastloss
				if(SAVE_FOLD):
					model_fold = SAVE_DATE+'_snd_kernel'+str(COV_KERNEL_SIZE[COV_i])+'_filter'+str(FILTER_NUM[filter_i])+'_fold'+str(fold)+'.pt'
					torch.save(res,model_fold)
					print('file saved, named:',model_fold)
			train_end = datetime.datetime.now()
			for fold in range(FOLD):
				print(fold_confusion[fold],end='	')
				print('Fold ',fold,' accuracy:%.4f' %fold_acc[fold])
			print(res)
			print('tag type:',DATA_TYPE)
			print('epoch:                  ',EPOCH)
			print('batch size:             ',BATCH_SIZE)
			print('learning rate:          ',LR)
			print('################################################################')
			train_time = train_end-train_start
			resulttxt = SAVE_DATE+'snd'+RESULT_FILE
			write_result(RESULT_PATH, resulttxt, fold_acc, fold_loss, fold_confusion, EPOCH, BATCH_SIZE, LR, DATA_TYPE, res, train_time)
			all_result.append([fold_acc, fold_loss, fold_confusion, res, train_time, FILTER_NUM[filter_i], COV_KERNEL_SIZE[COV_i]])
			print(np.mean(fold_acc))
			if np.mean(fold_acc) > BEST_ACC:
				BEST_ACC = np.mean(fold_acc)
				BEST_FILE = resulttxt
				print(BEST_ACC,' ',BEST_FILE)
			if np.mean(fold_loss) < BEST_LOSS:
				BEST_LOSS = np.mean(fold_loss)
				BEST_LFILE = resulttxt

	end_time = datetime.datetime.now()

	print('begin:',begin_time,'end:',end_time,'run time:',end_time- begin_time)
	print(BEST_FILE)
	print('best accuracy:',BEST_ACC)
	print(BEST_LFILE)
	print('best loss:',BEST_LOSS)

	if SAVE_BESTMODEL:
		for epoch in range(EPOCH):
			
			res.float().to(device)
			optimizer = torch.optim.Adam(res.parameters(),lr = LR)
			loss_func = nn.CrossEntropyLoss()
			res, lastloss = train_model(wavs, tags, BATCH_SIZE, res, optimizer, loss_func, device)
		model = SAVE_DATE+'_sndsilence_res_total.pt'
		torch.save(res,MODEL_PATH+model)
		print('file saved, named:',model)

	wf = open('./'+SAVE_DATE+'snd_best.txt','w')
	wf.write('maximum accuracy file:'+BEST_FILE)
	wf.write('\naccuracy: '+str(BEST_ACC))
	wf.write('\nminimum loss file:'+BEST_LFILE)
	wf.write('\nloss: '+str(BEST_LOSS))
	wf.write('\ntraining time:'+str(end_time-begin_time))
	wf.write('\n')
	for i in all_result:
		wf.write('*********************************\n')
		wf.write('filter: '+str(i[-2])+' kernel size: '+str(i[-1])+' train time:'+str(i[-3])+'\n')
		wf.write('accuracy: '+str(i[0])+'\n')
		wf.write('loss:     '+str(i[1])+'\n')
	wf.close()


if __name__ == "__main__":
    main()