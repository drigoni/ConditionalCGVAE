#!/usr/bin/env/python
"""
Usage:
    make_dataset.py [options]

Options:
    -h --help           Show this screen.
    --dataset NAME      QM9 or ZINC
"""

import json
import os
import sys

import numpy as np
from docopt import docopt
from rdkit import Chem
from rdkit.Chem import QED

import utils

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# get current directory in order to work with full path and not dynamic
current_dir = os.path.dirname(os.path.realpath(__file__))


def readStr_qm9():
    f = open(current_dir + '/qm9.smi', 'r')
    L = []
    for line in f:
        line = line.strip()
        L.append(line)
    f.close()
    np.random.seed(1)
    np.random.shuffle(L)
    return L


def read_zinc():
    f = open(current_dir + '/zinc.smi', 'r')
    L = []
    for line in f:
        line = line.strip()
        L.append(line)
    f.close()
    return L


def train_valid_split(dataset):
    n_mol_out = 0
    n_test = 5000
    test_idx = np.arange(0, n_test)
    valid_idx = np.random.randint(n_test, high=len(dataset), size=round(len(dataset) * 0.1))

    # save the train, valid dataset.
    raw_data = {'train': [], 'valid': [], 'test': []}
    file_count = 0
    for i, smiles in enumerate(dataset):
        val = QED.qed(Chem.MolFromSmiles(smiles))
        hist = make_hist(smiles)
        if hist is not None:
            if i in valid_idx:
                raw_data['valid'].append({'smiles': smiles, 'QED': val, 'hist': hist.tolist()})
            elif i in test_idx:
                raw_data['test'].append({'smiles': smiles, 'QED': val, 'hist': hist.tolist()})
            else:
                raw_data['train'].append({'smiles': smiles, 'QED': val, 'hist': hist.tolist()})
            file_count += 1
            if file_count % 1000 == 0:
                print('Finished reading: %d' % file_count, end='\r')
        else:
            n_mol_out += 1

    print("Number of molecules left out: ", n_mol_out)
    return raw_data


def make_hist(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()
    hist = np.zeros(utils.dataset_info(dataset)['hist_dim'])
    for atom in atoms:
        if dataset == 'qm9':
            atom_str = atom.GetSymbol()
        else:
            # zinc dataset # transform using "<atom_symbol><valence>(<charge>)"  notation
            symbol = atom.GetSymbol()
            valence = atom.GetTotalValence()
            charge = atom.GetFormalCharge()
            atom_str = "%s%i(%i)" % (symbol, valence, charge)

            if atom_str not in utils.dataset_info(dataset)['atom_types']:
                print('Unrecognized atom type %s' % atom_str)
                return None

        ind = utils.dataset_info(dataset)['atom_types'].index(atom_str)
        val = utils.dataset_info(dataset)['maximum_valence'][ind]
        hist[val - 1] += 1  # in the array the valence number start from 1, instead the array start from 0
    return hist


def preprocess(raw_data, dataset):
    print('Parsing smiles as graphs...')
    processed_data = {'train': [], 'valid': [], 'test': []}

    file_count = 0
    for section in ['train', 'valid', 'test']:
        all_smiles = []  # record all smiles in training dataset
        for i, (smiles, QED, hist) in enumerate([(mol['smiles'], mol['QED'], mol['hist'])
                                                 for mol in raw_data[section]]):
            nodes, edges = utils.to_graph(smiles, dataset)
            if len(edges) <= 0:
                print('Error. Molecule with len(edges) <= 0')
                continue
            processed_data[section].append({
                'targets': [[QED]],
                'graph': edges,
                'node_features': nodes,
                'smiles': smiles,
                'hist': hist
            })
            all_smiles.append(smiles)
            if file_count % 1000 == 0:
                print('Finished processing: %d' % file_count, end='\r')
            file_count += 1
        print('%s: 100 %%                   ' % (section))
        with open('molecules_%s_%s.json' % (section, dataset), 'w') as f:
            json.dump(processed_data[section], f)

    print("Train molecules = " + str(len(processed_data['train'])))
    print("Valid molecules = " + str(len(processed_data['valid'])))
    print("Test molecules = " + str(len(processed_data['test'])))


if __name__ == "__main__":
    args = docopt(__doc__)
    dataset = args.get('--dataset')

    print('Reading dataset: ' + str(dataset))
    data = []
    if dataset == 'qm9':
        data = readStr_qm9()
    elif dataset == 'zinc':
        data = read_zinc()
    else:
        print('Error. The database doesn\'t exist')
        exit(1)

    raw_data = train_valid_split(data)
    preprocess(raw_data, dataset)
