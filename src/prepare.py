# prepare.py
# Generate voxels on input PDB files
# Copyright (C) 2021 Linyuan Guo

# Codes for reading python files (line 72-140, 162-197, 219-302) were taken from data_processing/prepare_input.py (https://github.com/kiharalab/DOVE) under GNU General Public License version 3

# Copyright (C) 2019 Wang, Xiao and Terashi, Genki and Christoffer, Charles W and Zhu, Mengmeng and Kihara, Daisuke and Purdue University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import argparse, os, time, sys, gc
import src.AtomTypeDictionary as ATD


def Func_RotationX(vector, theta):
	RX = np.array([[1, 0, 0]
					, [0, np.cos(theta), -np.sin(theta)]
					, [0, np.sin(theta), np.cos(theta)]])
	return np.dot(RX, vector)


def Func_RotationY(vector, theta):
	RY = np.array([[np.cos(theta), 0, np.sin(theta)]
					, [0, 1, 0]
					, [-np.sin(theta), 0, np.cos(theta)]])
	return np.dot(RY, vector)


def Func_RotationZ(vector, theta):
	RZ = np.array([[np.cos(theta), -np.sin(theta), 0]
					, [np.sin(theta), np.cos(theta), 0]
					, [0, 0, 1]])
	return np.dot(RZ, vector)


def Func_ProteinRotation(R_InterfaceList, L_InterfaceList, axis, theta):
	for res in R_InterfaceList:
		for atom in res:
			v = atom[0:3]
			if axis == 'X' or axis == 'x':
				new_v = Func_RotationX(v, np.pi * theta / 180)
			elif axis == 'Y' or axis == 'y':
				new_v = Func_RotationY(v, np.pi * theta / 180)
			elif axis == 'Z' or axis == 'z':
				new_v = Func_RotationZ(v, np.pi * theta / 180)
			atom[0:3] = new_v
	for res in L_InterfaceList:
		for atom in res:
			v = atom[0:3]
			if axis == 'X' or axis == 'x':
				new_v = Func_RotationX(v, np.pi * theta / 180)
			elif axis == 'Y' or axis == 'y':
				new_v = Func_RotationY(v, np.pi * theta / 180)
			elif axis == 'Z' or axis == 'z':
				new_v = Func_RotationZ(v, np.pi * theta / 180)
			atom[0:3] = new_v
	return R_InterfaceList, L_InterfaceList


def form_atom_list(PDBLines):
	rlist = []
	llist = []
	rcount = 0
	for line in PDBLines:
		if line[0:4] == 'ATOM' and (line[21] == 'P' or line[21] == 'A'):
			rcount += 1
	atomid = 0
	count = 1
	goon = False
	residue_type = ''
	pre_residue_type = residue_type
	tmp_list = []
	pre_residue_id = 0
	pre_chain_id = ''
	for line in PDBLines:
		if len(line) == 0:
			continue
		if (line[0:4] == 'ATOM'):
			dat_in = line[0:80].split()
			chain_id = line[21]
			residue_id = int(line[22:26])
			if (atomid > int(dat_in[1])):
				if count <= rcount + 20 and count >= rcount - 20:
					goon = True
			if residue_id < pre_residue_id:
				if count <= rcount + 20 and count >= rcount - 20:
					goon = True
			if pre_chain_id != chain_id:
				if count <= rcount + 20 and count >= rcount - 20:
					goon = True
			x = float(line[30:38])
			y = float(line[38:46])
			z = float(line[46:54])
			residue_type = line[17:26]
			# First try CA distance of contact map
			atom_name = line[13:16].split()[0]
			residue_name = line[17:20].strip()
			chain_residue_id = chain_id + str(residue_id)
			atom_type = [atom_name, residue_name, chain_residue_id]
			if (goon):  # ligand list
				if pre_residue_type == residue_type:
					tmp_list.append([x, y, z, atom_type])
				else:
					if tmp_list:  # avoid empty list
						llist.append(tmp_list)
					tmp_list = []
					tmp_list.append([x, y, z, atom_type])
			else:  # receptor list
				if pre_residue_type == residue_type:
					tmp_list.append([x, y, z, atom_type])
				else:
					if tmp_list:  # avoid empty list
						rlist.append(tmp_list)
					tmp_list = []
					tmp_list.append([x, y, z, atom_type])
			atomid = int(dat_in[1])
			chainid = line[21]
			count = count + 1
			pre_residue_type = residue_type
			pre_residue_id = residue_id
			pre_chain_id = chain_id
	# print('in total, we have %d residues in receptor, %d residues in ligand' % (len(rlist), len(llist)))
	return rlist, llist


def form_interface(rlist, llist):
	"""
	:param rlist: receptor info
	:param llist: ligand info
	:param type: no use
	:return:
	information in the interface area
	"""
	# type=1:20A type=others:40A
	cut_off = 10
	cut_off = cut_off ** 2
	rindex_set = set()
	lindex_set = set()
	rindex = 0
	for r_item in rlist:
		lindex = 0
		for l_item in llist:
			min_distance = 1000000
			for r_atom in r_item:
				for l_atom in l_item:
					distance = \
						(r_atom[0] - l_atom[0]) ** 2 + (r_atom[1] - l_atom[1]) ** 2 + (r_atom[2] - l_atom[2]) ** 2
					if distance <= min_distance:
						min_distance = distance
			if min_distance <= cut_off:
				rindex_set.add(rindex)
				lindex_set.add(lindex)
			lindex += 1
		rindex += 1
	newrlist = []
	newllist = []
	for index in rindex_set:
		newrlist.append(rlist[index])
	for index in lindex_set:
		newllist.append(llist[index])
	return newrlist, newllist


def reform_input(rlist, llist, type=2):
	assert type == 1 or type == 2
	if type == 1:
		cut_off = 20
	else:
		cut_off = 40
	rlist1 = []  # form new rlist,(atom coordinate list instead of previous list)
	llist1 = []
	rlist2 = []  # Form new list, (atom list includes all information)
	llist2 = []
	for rindex, item1 in enumerate(rlist):
		residue1_len = len(item1)
		for i in range(residue1_len):
			atom = item1[i]
			tmp_list = []  # 记录原子x,y,z坐标
			for k in range(3):
				tmp_list.append(atom[k])
			rlist1.append(tmp_list)
			rlist2.append(atom[3])

	for lindex, item1 in enumerate(llist):
		residue1_len = len(item1)
		for i in range(residue1_len):
			atom = item1[i]
			tmp_list = []
			for k in range(3):
				tmp_list.append(atom[k])
			llist1.append(tmp_list)
			llist2.append(atom[3])
	rlist1 = np.array(rlist1)
	llist1 = np.array(llist1)
	rlist = rlist1
	llist = llist1
	coordinate = np.concatenate([rlist1, llist1])
	xmean = np.mean(coordinate[:, 0])
	ymean = np.mean(coordinate[:, 1])
	zmean = np.mean(coordinate[:, 2])
	half_cut = int(cut_off / 2)
	divide = 1
	if half_cut == 20:
		divide = 2
		half_cut = 10
	tempinput = np.zeros([20, 20, 20, 19, 4])
	count19_list = [0] * 19
	count_use = 0

	for i, item in enumerate(rlist):
		xo = int((float(item[0]) - xmean) / divide)
		yo = int((float(item[1]) - ymean) / divide)
		zo = int((float(item[2]) - zmean) / divide)
		Status = True
		atom_type = rlist2[i]
		if xo <= -half_cut or xo >= half_cut or yo <= -half_cut or yo >= half_cut or zo <= -half_cut or zo >= half_cut:
			Status = False

		if Status:
			count_use += 1
			channel = ATD.Func_GetAtomType19(atom_type[0], atom_type[1])
			tempinput[xo + half_cut, yo + half_cut, zo + half_cut, channel, 0] += 1
			tempinput[-xo + half_cut, -yo + half_cut, zo + half_cut, channel, 1] += 1
			tempinput[-yo + half_cut, xo + half_cut, zo + half_cut, channel, 2] += 1
			tempinput[yo + half_cut, -xo + half_cut, zo + half_cut, channel, 3] += 1
			count19_list[channel] += 1

	for i, item in enumerate(llist):
		xo = int((float(item[0]) - xmean) / divide)
		yo = int((float(item[1]) - ymean) / divide)
		zo = int((float(item[2]) - zmean) / divide)
		Status = True
		atom_type = llist2[i]
		if xo <= -half_cut or xo >= half_cut or yo <= -half_cut or yo >= half_cut or zo <= -half_cut or zo >= half_cut:
			Status = False
		if Status:
			count_use += 1
			channel = ATD.Func_GetAtomType19(atom_type[0], atom_type[1])
			tempinput[xo + half_cut, yo + half_cut, zo + half_cut, channel, 0] += 1
			tempinput[-xo + half_cut, -yo + half_cut, zo + half_cut, channel, 1] += 1
			tempinput[-yo + half_cut, xo + half_cut, zo + half_cut, channel, 2] += 1
			tempinput[yo + half_cut, -xo + half_cut, zo + half_cut, channel, 3] += 1
			count19_list[channel] += 1
	InputMatrixList = []
	for i in range(4):
		InputMatrixList.append(tempinput[:, :, :, :, i])
	return InputMatrixList


# list of PDB files in path
def Func_ListPDB(path):
	L = []
	for root, dirs, files in os.walk(path):
		for file in files:
			if os.path.splitext(file)[1].lower() == '.pdb':
				L.append(file)
	return L


def Func_SaveNPZ(strDecoysFolder, strOutputFolder):
	if not os.path.exists(strOutputFolder):
		os.mkdir(strOutputFolder)
	L = Func_ListPDB(strDecoysFolder)
	for p in L:
		file = open(os.path.join(strDecoysFolder, p))
		PDBLines = file.readlines()
		file.close()
		rlist, llist = form_atom_list(PDBLines)
		newrlist, newllist = form_interface(rlist, llist)
		InputMatrix = reform_input(newrlist, newllist)
		InputGrid = np.asarray(InputMatrix, dtype=np.uint8)
		np.savez_compressed(strOutputFolder + '/' + p + '.npz', InputGrid)


def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-F', type=str, required=True, help='path of saving PDB files')
	parser.add_argument('-O', type=str, required=True, help='path to saving NPZ files')
	args = parser.parse_args()
	params = vars(args)
	return params


if __name__ == '__main__':
	params = argparser()
	strDecoysFolder = params['F']
	strOutputFolder = params['O']
	Func_SaveNPZ(strDecoysFolder, strOutputFolder)
