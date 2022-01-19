import os, sys, traceback, gc, time, multiprocessing, shutil
import argparse
import numpy as np
# from model import *
from src.prepare import Func_SaveNPZ
from src.predict import *


def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-F', type=str, required=True, help='decoy example path of PDB files')
	parser.add_argument('-M', type=str, required=True, help='path of model (.pt)')
	parser.add_argument('-W', type=str, required=True, help='working path to save NPZ files')
	parser.add_argument('--gpu', type=str, default='0',
	                    help='Choose gpu id, example: \'0,1\'(specify use gpu 0, 1 or any other)')
	args = parser.parse_args()
	params = vars(args)
	return params


if __name__ == '__main__':
	params = argparser()
	strPDBFolder = params['F']
	strModelPath = params['M']
	strOutputFolder = params['W']
	Func_SaveNPZ(strDecoysFolder=strPDBFolder, strOutputFolder=strOutputFolder)
	listOfDecoys, listOfNPZ = Func_ReadNPZ(strNPZFolder=strOutputFolder)
	model = Func_LoadModel(strModelPath=strModelPath)
	results = Func_Predict(model, listOfDecoysNPZ=listOfNPZ)

	file = open(os.path.join(strOutputFolder, 'examples_results.txt'), 'w')
	for index, r in enumerate(results):
		file.write('%s\t%f\n' % (listOfDecoys[index], r))
	file.close()

	print('Finished ALL!')
