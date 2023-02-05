import torch
import torch.nn as nn
import torch.nn.functional as functional
########################### note ############################
# torch 0.4: https://pytorch.org/2018/04/22/0_4_0-migration-guide.html
# best sound avg pooling is 200(filter num) * 8(pool)
#############################################################

class Seq2Seq_attention_BLSTM(nn.Module):
	def __init__(self, input_dim = 1600, hidden_dim = 32, output_dim = 8):
		super(Seq2Seq_attention_BLSTM,self).__init__()
		
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.output_dim = output_dim
		
		self.encode_lstm = nn.LSTM(input_size = self.input_dim, 
							hidden_size = self.hidden_dim,
							num_layers = 1,
							bidirectional = True)
		self.fullconnect = nn.Linear(self.hidden_dim*2, self.hidden_dim)

		self.decode_lstm = nn.LSTM(input_size = self.hidden_dim,
							hidden_size  = self.hidden_dim,
							num_layers = 1,
							bidirectional = False)
		self.attention = nn.Linear(self.hidden_dim*2, 1)
		self.attention_combine = nn.Linear(self.hidden_dim*2, self.hidden_dim)
		self.outseq = nn.Linear(self.hidden_dim, self.output_dim)

		# self.hidden = self.init_hidden()
	
	# def init_hidden(self):
	# 	return(torch.zero(2,1,self.hidden_dim),torch.zero(1,1,self.hidden_dim))

	def forward(self, x, device):
		# encoder, give decode en_out, h_n
		en_out, (h_n, h_c) = self.encode_lstm(x,None) # None for all zero initial hidden state
		h_n = self.fullconnect(torch.cat((h_n[-1], h_n[-2]), dim=1).float())
		h_n = h_n.view(1, -1, self.hidden_dim)
		h_c = self.fullconnect(torch.cat((h_c[-1], h_c[-2]), dim=1).float())
		h_c = h_c.view(1, -1, self.hidden_dim)
		en_out_size = en_out.size()
		en_out = self.fullconnect(en_out.view(-1, en_out.size(2)).float())
		en_out = en_out.view(en_out_size[0], en_out_size[1], self.hidden_dim)

		# attention
		zero_input = torch.zeros(len(x),1,self.hidden_dim).to(device)
		attention_input = torch.cat((h_n.expand(len(x),1,self.hidden_dim),en_out),dim=2)
		weight = self.attention(attention_input.view(-1,self.hidden_dim*2))
		weight = weight.view(-1,1,1)
		weight = functional.softmax(weight, dim=0)
		applied = torch.bmm(weight, en_out)

		# decoder output sequence
		de_in = torch.cat((zero_input, applied), 2)
		de_in = self.attention_combine(de_in.view(-1,self.hidden_dim*2))
		de_in = de_in.view(-1,1,self.hidden_dim)
		de_in = functional.relu(de_in)
		de_out, de_h_n = self.decode_lstm(de_in, (h_n, h_c))
		de_out = self.outseq(de_out.view(-1,self.hidden_dim).view(-1,1,self.hidden_dim))
		
		return de_out, de_h_n, weight

