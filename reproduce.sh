python3 -m venv venv.
source venv./bin/activate
pip install gymnasium
pip install "gymnasium[box2d]"
pip install torch
pip install pyyaml
pip install "gymnasium[other]"
python3 project/eval.py --config config/test_eval.yaml


# python3 -m venv venv.
# pip install swig
# pip install gymnasium



# # gymnasium[box2d]
# # torch
# # pyyaml
# # moviepy

# Faire en une commande
# python project/eval.py --config config/test.yaml