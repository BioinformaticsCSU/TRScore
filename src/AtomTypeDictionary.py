AtomTypeDict = {
	'N': 0,
	'CA': 1,
	'C': 2,
	'O': 3,
	'CA-GLY': 4,
	'CB-ALA': 5, 'CB-ARG': 5, 'CB-ASN': 5, 'CB-ASP': 5, 'CB-CYS': 5, 'CB-CYX': 5, 'CB-GLN': 5, 'CB-GLU': 5,
	'CB-HIS': 5, 'CB-HID': 5, 'CB-HIE': 5, 'CB-HIP': 5,
	'CB-ILE': 5, 'CB-LEU': 5, 'CB-LYS': 5, 'CB-MET': 5, 'CB-PHE': 5, 'CB-PRO': 5, 'CG-PRO': 5, 'CD-PRO': 5,
	'CB-THR': 5, 'CB-TRP': 5, 'CB-TYR': 5, 'CB-VAL': 5,
	'CE-LYS': 6, 'NZ-LYS': 6,
	'CD-LYS': 7,
	'CG-ASP': 8, 'OD1-ASP': 8, 'OD2-ASP': 8, 'CD-GLU': 8, 'OE1-GLU': 8, 'OE2-GLU': 8,
	'CZ-ARG': 9, 'NH1-ARG': 9, 'NH2-ARG': 9,
	'CG-ASN': 10, 'OD1-ASN': 10, 'ND2-ASN': 10, 'CD-GLN': 10, 'OE1-GLN': 10, 'NE2-GLN': 10,
	'CD-ARG': 11, 'NE-ARG': 11,
	'CB-SER': 12, 'OG-SER': 12, 'OG1-THR': 12, 'OH-TYR': 12,
	'CG-HIS': 13, 'CG-HID': 13, 'CG-HIE': 13, 'CG-HIP': 13,
	'ND1-HIS': 13, 'ND1-HID': 13, 'ND1-HIE': 13, 'ND1-HIP': 13,
	'CD2-HIS': 13, 'CD2-HID': 13, 'CD2-HIE': 13, 'CD2-HIP': 13,
	'CE1-HIS': 13, 'CE1-HID': 13, 'CE1-HIE': 13, 'CE1-HIP': 13,
	'NE2-HIS': 13, 'NE2-HID': 13, 'NE2-HIE': 13, 'NE2-HIP': 13,
	'NE1-TRP': 13,
	'CE1-TYR': 14, 'CE2-TYR': 14, 'CZ-TYR': 14,
	'CG-ARG': 15, 'CG-GLN': 15, 'CG-GLU': 15, 'CG1-ILE': 15, 'CG-LEU': 15, 'CG-LYS': 15, 'CG-MET': 15, 'SD-MET': 15,
	'CG-PHE': 15, 'CD1-PHE': 15, 'CD2-PHE': 15, 'CE1-PHE': 15, 'CE2-PHE': 15, 'CZ-PHE': 15, 'CG2-THR': 15,
	'CG-TRP': 15, 'CD1-TRP': 15, 'CD2-TRP': 15, 'CE2-TRP': 15, 'CE3-TRP': 15, 'CZ2-TRP': 15, 'CZ3-TRP': 15,
	'CH2-TRP': 15,
	'CG-TYR': 15, 'CD1-TYR': 15, 'CD2-TYR': 15,
	'CG2-ILE': 16, 'CD1-ILE': 16, 'CD1-LEU': 16, 'CD2-LEU': 16, 'CE-MET': 16, 'CG1-VAL': 16, 'CG2-VAL': 16,
	'SG-CYS': 17, 'SG-CYX': 17
}


def Func_GetAtomType19(AtomName, ResName):
	try:
		if ResName == 'GLY':
			return 4
		elif AtomName in AtomTypeDict:
			return AtomTypeDict[AtomName]
		else:
			return AtomTypeDict[AtomName + '-' + ResName]
	except:
		return 18
