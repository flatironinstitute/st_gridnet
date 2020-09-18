import sys, os
import argparse
from matplotlib import pyplot as plt

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("../..")
from src.datasets import PatchGridDataset, StitchGridDataset
from src.densenet import DenseNet
from src.resnet import resnetseg34
from src.gridnet_patches import GridNet
from src.utils import all_fgd_predictions, class_auroc, plot_confusion_matrix
from src.utils import misclass_density, plot_class_boundaries

def aba(img_test, lbl_test, model_files, resnetseg):
	class_names = ["Midbrain","Isocortex","Medulla","Striatum",
		"C. nuclei","C. cortex","Thalamus","Olf. areas",
		"Cort. sub.","Pons","Pallidum","Hipp. form.","Hypothal."]

	patch_size = 256
	h_st, w_st = 32, 49
	xform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(patch_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	if resnetseg:
		test_set = StitchGridDataset(img_test, lbl_test, xform)
	else:
		test_set = PatchGridDataset(img_test, lbl_test, xform)

	return test_set, class_names, model_files, patch_size, h_st, w_st

def maniatis(img_test, lbl_test, model_files, resnetseg):
	class_names = ["Vent. Med. White", "Vent. Horn", "Vent. Lat. White", "Med. Gray", 
		"Dors. Horn", "Dors. Edge", "Med. Lat. White", "Vent. Edge", "Dors. Med. White",
		"Cent. Canal", "Lat. Edge"]

	patch_size = 256
	h_st, w_st = 35, 33
	xform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(patch_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	if resnetseg:
		test_set = StitchGridDataset(img_test, lbl_test, xform)
	else:
		test_set = PatchGridDataset(img_test, lbl_test, xform)

	return test_set, class_names, model_files, patch_size, h_st, w_st

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help='"aba" or "maniatis"')
	parser.add_argument('imgtest', type=str, help='Path to directory of test set image data')
	parser.add_argument('lbltest', type=str, help='Path to directory of test set label data')
	parser.add_argument('-m', '--model-files', nargs="+", required=True, help='List of model file paths')
	parser.add_argument("-r", "--resnetseg", action="store_true", help='Use ResNet-34-Seg model')
	parser.add_argument('-i', '--index', type=int, default=0, help='Index of image to use for misclassification density plot')
	args = parser.parse_args()

	if args.dataset == "aba":
		data_fn = aba
	else:
		data_fn = maniatis
	test_set, class_names, model_files, patch_size, h_st, w_st = data_fn(
		args.imgtest, args.lbltest, args.model_files, args.resnetseg)

	# Loop through models and store results of evaluation
	all_accs, all_aurocs, all_smax = [], [], []
	all_g_smax = []
	all_patch_accs, all_patch_aurocs, all_patch_smax = [], [], []

	for mf in model_files:
		dl = DataLoader(test_set, batch_size=1, shuffle=False)

		n_class = len(class_names)

		# Evaluate resnet segmentation baseline
		if args.resnetseg:
			base = "resnetseg_%s" % args.dataset
			g = resnetseg34(n_class, (h_st, w_st), thin=4)
		# Evaluate GridNet
		else:
			base = args.dataset
			f = DenseNet(num_classes=n_class, small_inputs=False, efficient=True,
				growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, 
				bn_size=4, drop_rate=0)
			g = GridNet(f, patch_shape=(3,patch_size,patch_size), grid_shape=(h_st, w_st), 
				n_classes=n_class, use_bn=False, atonce_patch_limit=300)
		if torch.cuda.is_available():
			g.load_state_dict(torch.load(mf))
		else:
			g.load_state_dict(torch.load(mf, map_location=torch.device('cpu')))
		g.eval()

		# For misclass density plot
		x,y = test_set[args.index] # L7CN32_C2
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		g = g.to(device)
		x = x.to(device)
		g_raw = g(x)[0]
		g_smax = F.softmax(g_raw, dim=0).cpu()
		all_g_smax.append(g_smax.data.numpy())
		
		# Calculate test accuracy, AUROC for full model
		true, pred, smax = all_fgd_predictions(dl, g)
		all_smax.append(smax)

		auroc = class_auroc(smax, true)
		all_aurocs.append(auroc)
		print('AUROC: {}'.format("\t".join(list(map(str, auroc)))), flush=True)

		test_acc = float(sum(true==pred))/len(true)
		all_accs.append(test_acc)
		print("Test Set Accuracy: %.4f" % test_acc, flush=True)
		
		# ...if GridNet, compare against patch classifier alone
		if not args.resnetseg:
			true, pred, smax = all_fgd_predictions(dl, g, f_only=True)
			all_patch_smax.append(smax)

			auroc = class_auroc(smax, true)
			all_patch_aurocs.append(auroc)
			print('Patch AUROC: {}'.format("\t".join(list(map(str, auroc)))), flush=True)

			test_acc = float(sum(true==pred))/len(true)
			all_patch_accs.append(test_acc)
			print("Patch Test Set Accuracy: %.4f" % test_acc, flush=True)

	all_aurocs = np.array(all_aurocs)
	all_accs = np.array(all_accs)
	all_smax = np.array(all_smax)

	# Misclass density plot
	mean_g_smax = np.array(all_g_smax).mean(axis=0)
	g_mcd = misclass_density(mean_g_smax, y.data.numpy())
	g_fig = plot_class_boundaries(g_mcd, y.data.numpy())
	plt.savefig("%s_misclass_density_consensus_g.png" % base, format="PNG", dpi=300)

	# Mean, STD, and ensemble performance:
	print("")
	print('Mean AUROC: {}'.format("\t".join(list(map(str, all_aurocs.mean(axis=0))))), flush=True)
	print('SD(AUROC): {}'.format("\t".join(list(map(str, all_aurocs.std(axis=0))))), flush=True)
	print('Macro Avg: %.4f (%.4f)' % (all_aurocs.mean(), all_aurocs.std()))

	print("")
	print("Mean Accuracy: %.4f" % all_accs.mean(axis=0), flush=True)
	print("SD(Accuracy): %.4f" % all_accs.std(axis=0), flush=True)

	print("")
	consensus_smax = np.array(all_smax).mean(axis=0)
	consensus_auroc = class_auroc(consensus_smax, true)
	print("Consensus AUROC: {}".format("\t".join(list(map(str, consensus_auroc)))), flush=True)
	print("Macro Avg: %.4f" % (consensus_auroc.mean()))
	consensus_pred = consensus_smax.argmax(axis=1)
	n_correct = np.sum(consensus_pred==true)
	print("Consensus Accuracy: %.4f" % (float(n_correct)/len(true)))

	if not args.resnetseg:
		all_patch_aurocs = np.array(all_patch_aurocs)
		all_patch_accs = np.array(all_patch_accs)
		all_patch_smax = np.array(all_patch_smax)

		print("")
		print('Mean Patch AUROC: {}'.format("\t".join(list(map(str, all_patch_aurocs.mean(axis=0))))), flush=True)
		print('SD(Patch AUROC): {}'.format("\t".join(list(map(str, all_patch_aurocs.std(axis=0))))), flush=True)
		print('Macro Avg: %.4f (%.4f)' % (all_patch_aurocs.mean(), all_patch_aurocs.std()))

		print("")
		print("Mean Patch Accuracy: %.4f" % all_patch_accs.mean(axis=0), flush=True)
		print("SD(Patch Accuracy): %.4f" % all_patch_accs.std(axis=0), flush=True)

		print("")
		consensus_patch_smax = np.array(all_patch_smax).mean(axis=0)
		consensus_patch_auroc = class_auroc(consensus_patch_smax, true)
		print("Consensus Patch AUROC: {}".format("\t".join(list(map(str, consensus_patch_auroc)))), flush=True)
		print("Macro Avg: %.4f" % (consensus_patch_auroc.mean()))
		consensus_patch_pred = consensus_patch_smax.argmax(axis=1)
		n_correct = np.sum(consensus_patch_pred==true)
		print("Consensus Patch Accuracy: %.4f" % (float(n_correct)/len(true)))

		

