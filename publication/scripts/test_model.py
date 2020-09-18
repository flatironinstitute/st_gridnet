import sys, os
import argparse
from matplotlib import pyplot as plt

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

def aba(img_test, lbl_test, model_file, resnetseg):
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

	return test_set, class_names, model_file, patch_size, h_st, w_st

def maniatis(img_test, lbl_test, model_file, resnetseg):
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

	return test_set, class_names, model_file, patch_size, h_st, w_st

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help='"aba" or "maniatis"')
	parser.add_argument('imgtest', type=str, help='Path to test set image directory')
	parser.add_argument('lbltest', type=str, help='Path to test set label directory')
	parser.add_argument('modelfile', type=str, help='Path to model file')
	parser.add_argument("-r", "--resnetseg", action="store_true")
	args = parser.parse_args()

	if args.dataset == "aba":
		data_fn = aba
	else:
		data_fn = maniatis
	test_set, class_names, model_file, patch_size, h_st, w_st = data_fn(
		args.imgtest, args.lbltest, args.modelfile, args.resnetseg)

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
			growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0)
		g = GridNet(f, patch_shape=(3,patch_size,patch_size), grid_shape=(h_st, w_st), n_classes=n_class, 
			use_bn=False, atonce_patch_limit=300)
	if torch.cuda.is_available():
		g.load_state_dict(torch.load(model_file))
	else:
		g.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
	g.eval()
	
	# Calculate test accuracy, AUROC for full model
	true, pred, smax = all_fgd_predictions(dl, g)

	cmat_freq = plot_confusion_matrix(true, pred, class_names, density=True)
	plt.savefig("../figures/%s_cmat_freq.png" % base, format="PNG")

	auroc = class_auroc(smax, true)
	print('AUROC: {}'.format("\t".join(list(map(str, auroc)))), flush=True)

	test_acc = float(sum(true==pred))/len(true)
	print("Test Set Accuracy: %.4f" % test_acc, flush=True)

	
	# ...if GridNet, compare against patch classifier alone
	if not args.resnetseg:
		true, pred, smax = all_fgd_predictions(dl, g, f_only=True)

		auroc = class_auroc(smax, true)
		print('Patch AUROC: {}'.format("\t".join(list(map(str, auroc)))), flush=True)

		test_acc = float(sum(true==pred))/len(true)
		print("Patch Test Set Accuracy: %.4f" % test_acc, flush=True)

	
	# Generate misclassification density plots for f, g
	if not args.resnetseg:
		if args.dataset == "maniatis":
			x,y = test_set[20] # L7CN32_C2
		else:
			x,y = test_set[29] # 102117900

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		g.to(device)
		x.to(device)

		f_raw = g.patch_predictions(x)[0]
		f_smax = F.softmax(f_raw, dim=0)

		g_raw = g(x)[0]
		g_smax = F.softmax(g_raw, dim=0)

		f_mcd = misclass_density(f_smax.data.numpy(), y.data.numpy())
		g_mcd = misclass_density(g_smax.data.numpy(), y.data.numpy())

		f_fig = plot_class_boundaries(f_mcd, y.data.numpy())
		plt.tight_layout()
		plt.savefig("../figures/%s_misclass_density_f.png" % base, format="PNG", dpi=300)

		g_fig = plot_class_boundaries(g_mcd, y.data.numpy())
		plt.savefig("../figures/%s_misclass_density_g.png" % base, format="PNG", dpi=300)

