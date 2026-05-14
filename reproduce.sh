#!/usr/bin/env bash

RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

echo -e     "${GREEN}Installation des dépendances..${RESET}"
python3 -m venv venv.
source venv./bin/activate
venv./bin/pip install gymnasium
venv./bin/pip install "gymnasium[box2d]"
venv./bin/pip install torch
venv./bin/pip install pyyaml
venv./bin/pip install "gymnasium[other]"

echo -e "${GREEN}Exécution du programme..${RESET}"
echo -e "${GREEN}Choisit la config que tu veux utiliser parmi la liste suivante :${RESET}"
ls config
echo -e "${GREEN}Indique le nom de la config (sans le ".yaml") :${RESET}"
read VARIABLE
case $VARIABLE in
    buffer_1K)
        echo "> buffer_1K <"
        python3 project/eval.py --config config/buffer_1K.yaml
        ;;
    buffer_10K)
        echo "> buffer_10K <"
        python3 project/eval.py --config config/buffer_10K.yaml
        ;;
    buffer_50K)
        echo "> buffer_50K <"
        python3 project/eval.py --config config/buffer_50K.yaml
        ;;
    buffer_90K)
        echo "> buffer_90K <"
        python3 project/eval.py --config config/buffer_90K.yaml
        ;;
    eval_buffer_50K_seed4)
        echo "> eval_buffer_50K_seed4 <"
        python3 project/eval.py --config config/eval_buffer_50K_seed4.yaml
        ;;
    eval_buffer_90K_seed0)
        echo "> eval_buffer_90K_seed0 <"
        python3 project/eval.py --config config/eval_buffer_90K_seed0.yaml
        ;;
    eval_buffer_90K_seed1)
        echo "> eval_buffer_90K_seed1 <"
        python3 project/eval.py --config config/eval_buffer_90K_seed1.yaml
        ;;
    eval_buffer_90K_seed2)
        echo "> eval_buffer_90K_seed2 <"
        python3 project/eval.py --config config/eval_buffer_90K_seed2.yaml
        ;;
    eval_buffer_90K_seed3)
        echo "> eval_buffer_90K_seed3 <"
        python3 project/eval.py --config config/eval_buffer_90K_seed3.yaml
        ;;
    test_eval)
        echo "> test_eval <"
        python3 project/eval.py --config config/test_eval.yaml
        ;;
    test)
        echo "> test <"
        python3 project/eval.py --config config/test.yaml
        ;;
    *)
        echo "${RED}Fichier inexistant${RESET}"
        ;;
esac
