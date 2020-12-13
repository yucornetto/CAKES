import argparse
import os
import sys
import time
import shutil
import torch
import torch.nn as nn
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm

from dataset import VideoDataSet
from models import TemporalModel
from transforms import *
from opts import parser
import datasets_video
from utils import WarmUpLR
from operations import *
from config import *

best_prec1 = 0

def main():
	global args, best_prec1
	args = parser.parse_args()
	print(args)
	check_rootfolders()

	categories, train_list, val_list, root_path, prefix = datasets_video.return_dataset(args.dataset,args.root_path)
	num_class = len(categories)


	global store_name 
	store_name = '_'.join([args.type, args.dataset, args.arch, 'segment%d'% args.num_segments, args.store_name])
	print(('storing name: ' + store_name))

	if args.dataset == 'somethingv1' or args.dataset == 'somethingv2':
		# label transformation for left/right categories
		# please refer to labels.json file in sometingv2 for detail.
		target_transforms = {86:87,87:86,93:94,94:93,166:167,167:166}
	else:
		target_transforms = None

	if args.conv_config in conv_configs:
		conv_config = conv_configs[args.conv_config]
		conv_index = None # conv_indexs[args.conv_config]
	else:
		conv_config = None

	model = TemporalModel(num_class, args.num_segments, model=args.type, backbone=args.arch,
						  alpha=args.alpha, beta=args.beta, dropout=args.dropout,
						  target_transforms=target_transforms, search=args.search, op_code=args.op_code, conv_config=conv_config)

	crop_size = model.crop_size
	scale_size = model.scale_size
	input_mean = model.input_mean
	input_std = model.input_std
	policies = get_optim_policies(model)
	train_augmentation = model.get_augmentation()


	if torch.cuda.is_available():
		model = torch.nn.DataParallel(model).cuda()

	
	if args.prune:
		prune(model, args.prune_model_path, './conv_config.txt')
		# prune_select(model, args.prune_model_path, './conv_config.txt')
		exit(0)


	if args.resume:
		if os.path.isfile(args.resume):
			print(("=> loading checkpoint '{}'".format(args.resume)))
			checkpoint = torch.load(args.resume)
			
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			# model.module.load_state_dict(checkpoint['state_dict'])
			model.load_state_dict(checkpoint['state_dict'])
			# print(("=> loaded checkpoint '{}' (epoch {})"
			# 	  	.format(args.evaluate, checkpoint['epoch'])))
		else:
			print(("=> no checkpoint found at '{}'".format(args.resume)))

	if args.finetune:
		if os.path.isfile(args.finetune):
			print(("=> loading checkpoint '{}'".format(args.finetune)))
			checkpoint = torch.load(args.finetune)
			from I3D import load_state_dict_supernet
			model = load_state_dict_supernet(model, checkpoint['state_dict'], conv_index)
			#args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			#model.module.load_state_dict(checkpoint['state_dict'])
			print(("=> loaded checkpoint '{}' (epoch {})"
				  	.format(args.evaluate, checkpoint['epoch'])))
		else:
			print(("=> no checkpoint found at '{}'".format(args.finetune)))
			exit(0)

	cudnn.benchmark = True

	# Data loading code
	normalize = GroupNormalize(input_mean, input_std)
	
	train_loader = torch.utils.data.DataLoader(
		VideoDataSet(root_path, train_list, num_segments=args.num_segments,
				   image_tmpl=prefix,
				   transform=torchvision.transforms.Compose([
					   train_augmentation,
					   Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
					   ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
					   normalize,
				   ])),
		batch_size=args.batch_size, shuffle=True, drop_last=True,
		num_workers=args.workers, pin_memory=True)

	val_loader = torch.utils.data.DataLoader(
		VideoDataSet(root_path, val_list, num_segments=args.num_segments,
				   image_tmpl=prefix,
				   random_shift=False,
				   transform=torchvision.transforms.Compose([
					   GroupScale(int(scale_size)),
					   GroupCenterCrop(crop_size),
					   Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
					   ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
					   normalize,
				   ])),
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	# define loss function (criterion) and optimizer
	criterion = torch.nn.CrossEntropyLoss().cuda()
	
	for group in policies:
		print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
			group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

	optimizer = torch.optim.SGD(policies,
								args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)


	if args.evaluate:
		prec1 = validate(val_loader, model, criterion, 0)
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		# save_checkpoint({
		# 		'epoch': args.start_epoch,
		# 		'arch': args.arch,
		# 		'state_dict': model.state_dict(),
		# 		'best_prec1': best_prec1,
		# 	}, is_best)
		exit()

		return

	log_training = open(os.path.join(args.checkpoint_dir,'log', '%s.csv' % store_name), 'w')
	warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * args.warm)
	for epoch in range(args.start_epoch, args.epochs):
		# adjust learning rate
		if epoch > args.warm:
			adjust_learning_rate(optimizer, epoch, args.lr_steps)
		
		# train for one epoch
		train(train_loader, model, criterion, optimizer, epoch, log_training, warmup_scheduler, args)
		
		# evaluate on validation set
		if ((epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1):
			if 'dropout' not in args.op_code:
				prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader),
								 log_training)
			else:
				prec1 = 0.0

			# remember best prec@1 and save checkpoint
			is_best = prec1 > best_prec1
			best_prec1 = max(prec1, best_prec1)
			save_checkpoint({
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				'best_prec1': best_prec1,
			}, is_best)

def prune(model, weight_path, log_path):
	log = open(log_path, 'w')
	checkpoint = torch.load(weight_path)
	model.load_state_dict(checkpoint['state_dict'])
	new_conv_config = []
	for k, m in enumerate(model.modules()):
		if isinstance(m, ConvST) or isinstance(m, MixConv1D) or isinstance(m, MixConv2D) or isinstance(m, Conv1_2_3D) or isinstance(m, Conv1_2D) or isinstance(m, Conv2_3D) or isinstance(m, Conv1_2_3D_Dropout) or isinstance(m, Conv1_2D_Dropout) or isinstance(m, ConvST_Dropout):
			conv_config = m.pruneBN()
			new_conv_config.append(conv_config)

	print('conv config after pruning:', new_conv_config)
	log.write(str(new_conv_config).replace('tensor(', '').replace(", device='cuda:0')", ""))
	log.close()

def prune_select(model, weight_path, log_path):
	log = open(log_path, 'w')
	log_keep_index = open(log_path.replace('conv_config', 'conv_config_index'), 'w')
	checkpoint = torch.load(weight_path)
	model.load_state_dict(checkpoint['state_dict'])
	new_conv_config = []
	new_keep_index = []
	for k, m in enumerate(model.modules()):
		if isinstance(m, ConvST) or isinstance(m, MixConv1D) or isinstance(m, MixConv2D) or isinstance(m, Conv1_2_3D) or isinstance(m, Conv1_2D) or isinstance(m, Conv2_3D):
			conv_config, keep_index = m.pruneBN_select()
			new_conv_config.append(conv_config)
			new_keep_index.append(keep_index)

	print('keep index:', new_keep_index)
	print('conv config after pruning:', new_conv_config)
	log.write(str(new_conv_config).replace('tensor(', '').replace(", device='cuda:0')", ""))
	log_keep_index.write(str(new_keep_index).replace('tensor(', '').replace(", device='cuda:0')", ""))
	log.close()
	log_keep_index.close()

def train(train_loader, model, criterion, optimizer, epoch, log, warmup_scheduler, args):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	
	for i, (input, target) in enumerate(train_loader):
		if epoch < args.warm:
			warmup_scheduler.step()

		# measure data loading time
		data_time.update(time.time() - end)
		input = input.cuda(non_blocking = True)
		target = target.cuda(non_blocking=True)
		
		# compute output
		output = model(input)
		loss = criterion(output, target)

		# measure accuracy and record loss
		prec1, prec5 = accuracy(output, target, topk=(1,5))
		losses.update(loss.item(), input.size(0))
		top1.update(prec1[0], input.size(0))
		top5.update(prec5[0], input.size(0))


		# compute gradient and do SGD step
		optimizer.zero_grad()

		loss.backward()

		### l1 norm
		if args.sr:
			for m in model.modules():
				if isinstance(m, PruneBN):
					if args.reweight:
						#fenmu =  ### 3:1
						if args.op_code == 'convst' or args.op_code == 'conv1_2d':
							c = m.weight.data.shape[0] // 2 ### convst
							m.weight.grad[:c].data.add_( 3/4 * args.s * torch.sign(m.weight.data[:c]))
							m.weight.grad[c:].data.add_( 1/4 * args.s * torch.sign(m.weight.data[c:]))
						elif args.op_code == 'conv1_2_3d':
							# fenmu = ### 9:3:1
							c = m.weight.data.shape[0] // 3 ### convst
							m.weight.grad[:c].data.add_( 9/13 * args.s * torch.sign(m.weight.data[:c]))
							m.weight.grad[c:2*c].data.add_( 3/13 * args.s * torch.sign(m.weight.data[c:2*c]))
							m.weight.grad[2*c:3*c].data.add_( 1/13 * args.s * torch.sign(m.weight.data[2*c:3*c]))
					else:
						m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))

		if args.clip_gradient is not None:
			total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
			if total_norm > args.clip_gradient:
				print(("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm)))

		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
					'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
						epoch, i, len(train_loader), batch_time=batch_time,
						data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
			print(output)
			log.write(output + '\n')
			log.flush()



def validate(val_loader, model, criterion, iter, log = None):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()
	total = 0
	end = time.time()
	with torch.no_grad():
		for i, (input, target) in enumerate(val_loader):
			input = input.cuda(non_blocking = True)
			target = target.cuda(non_blocking = True)
			# compute output
			output = model(input)
			loss = criterion(output, target)

			# measure accuracy and record loss
			prec1, prec5 = accuracy(output, target, topk=(1,5))

			losses.update(loss.item(), input.size(0))
			top1.update(prec1[0], input.size(0))
			top5.update(prec5[0], input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				output = ('Test: [{0}/{1}]\t'
				 	 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				 	 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				 	 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				 	 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				 	  i, len(val_loader), batch_time=batch_time, loss=losses,
				  	 top1=top1, top5=top5))
				print(output)
				if log is not None:
					log.write(output + '\n')
					log.flush()

	output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
		  .format(top1=top1, top5=top5, loss=losses))
	print(output)
	output_best = '\nBest Prec@1: %.3f'%(best_prec1)
	print(output_best)
	if log is not None:
		log.write(output + ' ' + output_best + '\n')
		log.flush()

	return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	epoch = state['epoch']
	torch.save(state, '%s/%s_checkpoint.pth.tar' % (os.path.join(args.checkpoint_dir,'checkpoint'), store_name))
	if is_best:
		shutil.copyfile('%s/%s_checkpoint.pth.tar' % (os.path.join(args.checkpoint_dir,'checkpoint'), store_name),'%s/%s_best.pth.tar' % (os.path.join(args.checkpoint_dir,'checkpoint'), store_name))
	if epoch % 5 == 0:
		shutil.copyfile('%s/%s_checkpoint.pth.tar' % (
		os.path.join(args.checkpoint_dir, 'checkpoint'), store_name), '%s/%s_epoch%d.pth.tar' % (
						os.path.join(args.checkpoint_dir, 'checkpoint'), store_name, epoch))

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
	"""Sets the learning rate to the initial LR decayed by 10 """
	decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
	lr = args.lr * decay
	decay = args.weight_decay
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr * param_group['lr_mult']
		param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)
		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0,keepdim = True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

def check_rootfolders():
	"""Create log and model folder"""
	folders_util = [args.checkpoint_dir,os.path.join(args.checkpoint_dir,'log'), os.path.join(args.checkpoint_dir,'checkpoint')]
	for folder in folders_util:
		if not os.path.exists(folder):
			print(('creating folder ' + folder))
			os.makedirs(folder)

def get_optim_policies(model):
	first_conv_weight = []
	first_conv_bias = []
	normal_weight = []
	normal_bias = []
	bn = []

	conv_cnt = 0
	bn_cnt = 0
	for m in model.modules():
		if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d):
			ps = list(m.parameters())
			conv_cnt += 1
			if conv_cnt <= 3:
				first_conv_weight.append(ps[0])
				if len(ps) == 2:
					first_conv_bias.append(ps[1])
			else:
				normal_weight.append(ps[0])
				if len(ps) == 2:
					normal_bias.append(ps[1])
		
		elif isinstance(m, torch.nn.Linear):
			ps = list(m.parameters())
			normal_weight.append(ps[0])
			if len(ps) == 2:
				normal_bias.append(ps[1])

		elif isinstance(m,nn.BatchNorm3d):
			bn_cnt += 1
			bn.extend(list(m.parameters()))
		
		elif len(m._modules) == 0:
			if len(list(m.parameters())) > 0:
				raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
	return [
			{'params': first_conv_weight, 'lr_mult':  1, 'decay_mult': 1,
			 'name': "first_conv_weight"},
			{'params': first_conv_bias, 'lr_mult':  2, 'decay_mult': 0,
			 'name': "first_conv_bias"},
			{'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
			 'name': "normal_weight"},
			{'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
			 'name': "normal_bias"},
			{'params': bn, 'lr_mult': 1, 'decay_mult': 0,
			 'name': "BN scale/shift"},
		]



if __name__ == '__main__':
	main()
