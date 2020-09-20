#!/usr/bin/env bash

s1=$1
s2="install"
s3="remove"
s4="pretrained"

if [[ $s1 == $s2 ]]; then
    echo "-------------------------------------------------------------------------------------------------------------"
    echo "---------------------------------------------------  DATA - -------------------------------------------------"
    echo "-------------------------------------------------------------------------------------------------------------"
    cd data
    wget --no-check-certificate -O "qm9.zip" https://www.dropbox.com/s/cfra3r50j89863x/qm9.zip?dl=0
    unzip -a qm9.zip
    rm qm9.zip

    wget --no-check-certificate -O "zinc.zip" https://www.dropbox.com/s/rrisjasazovyouf/zinc.zip?dl=0
    unzip -a zinc.zip
    rm zinc.zip


    echo "-------------------------------------------------------------------------------------------------------------"
    echo "---------------------------------------------------  UTILS - ------------------------------------------------"
    echo "-------------------------------------------------------------------------------------------------------------"
    cd ../utils
    wget --no-check-certificate https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py
    wget --no-check-certificate https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz

    echo "------------------------------------------------------------------------------------------------------------"
    echo "---------------------------------------------------  CCGVVAE  -----------------------------------------------"
    echo "------------------------------------------------------------------------------------------------------------"
    conda env create -f ccgvae_env.yml
    conda activate ccgvae
    pip install Cython --install-option="--no-cython-compile"
    pip install -r ccgvae_env_requirements.txt
    conda deactivate

else
   if [[ $s1 == $s3 ]]; then
        conda deactivate
        conda remove -n givae --all
   else
   		echo "To be implemented."
        else
           echo Use "install", "remove" or "pretrained"
        fi
   fi
fi
