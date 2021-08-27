import os, torch
import numpy as np
from src.BuildModel import *


def Func_ListNPZ(path):
	L = []
	for root, dirs, files in os.walk(path):
		for file in files:
			if os.path.splitext(file)[1].lower() == '.npz':
				L.append(file)
	return L


def Func_ReadNPZ(strNPZFolder):
	L = Func_ListNPZ(strNPZFolder)
	listOfNPZ = []
	for p in L:
		f = os.path.join(strNPZFolder, p)
		with np.load(f) as m:
			listOfNPZ.append(np.asarray(m['arr_0'], dtype=np.uint8))
	return L, listOfNPZ


def Func_LoadModel(strModelPath):
	model = create_RepVGG(2, 19).cuda()
	model.load_state_dict(torch.load(strModelPath))
	model.eval()
	model = repvgg_model_convert(model)
	return model


def Func_Predict_single(model, NPZFile):
	return model(NPZFile)


def Func_Predict(model, listOfDecoysNPZ):
	results = []
	for m in listOfDecoysNPZ:
		m2 = np.transpose(m, (0, 4, 1, 2, 3))
		t = torch.Tensor(m2)
		r = model(t.cuda())
		r = torch.sigmoid(r).data
		results.append(torch.max(r).cpu())
	return results
