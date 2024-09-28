Running Instructions :

For Windows :

1. Using Conda environment
   Make sure that conda is installed in the local system.
   Create a Conda Environment. In the terminal, conda create --name virtualenv
   Activating the conda environment - 1. In the command Pallete, select Python: Select Interpreter -> select virtualenv conda environment
   OR 2.In the terminal, conda activte virtualenv
   In the terminal:
   For CPU : conda install pytorch torchvision cpuonly -c pytorch
   For GPU : conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
   Then clone the git repository - git clone https://github.com/ByteMeEthos/CCTV_footage_detection.git
   cd into the git directory - cd CCTV_footage_detection
   Installing dependencies - pip install -r requirements.txt
   cd into the inference folder - cd inference
   Finally for the output - python run_inference.py

See the model results in data -> output directory

2. Using python virtual environment
   Create a virtual environment - python -m venv virtualenv
   Activate the virtual environment - .\virtualenv\Scripts\activate
   Clone the git repository - git clone https://github.com/ByteMeEthos/CCTV_footage_detection.git
   cd into the git directory - cd CCTV_footage_detection
   Installing dependencies - pip install -r requirements.txt
   cd into the inference folder - cd inference
   Finally for the output - python run_inference.py

See the model results in data -> output directory

CCTV Footage Detection - Installation and Running Instructions
Installation Process Overview
mermaidCopygraph TD
A[Start] --> B[Choose Environment]
B --> C[Conda]
B --> D[Python venv]
C --> E[Create Conda Environment]
D --> F[Create Python venv]
E --> G[Activate Environment]
F --> G
G --> H[Install Dependencies]
H --> I[Clone Repository]
I --> J[Run Inference]
J --> K[View Results]
K --> L[End]
Windows Instructions

1. Using Conda Environment

Ensure Conda is installed on your system.
Create a Conda Environment:
bashCopyconda create --name virtualenv

Activate the Conda environment:

Option 1: In VS Code Command Palette, select Python: Select Interpreter -> select virtualenv conda environment
Option 2: In the terminal:
bashCopyconda activate virtualenv

Install PyTorch:

For CPU:
bashCopyconda install pytorch torchvision cpuonly -c pytorch

For GPU:
bashCopyconda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

Clone the repository:
bashCopygit clone https://github.com/ByteMeEthos/CCTV_footage_detection.git

Navigate to the project directory:
bashCopycd CCTV_footage_detection

Install dependencies:
bashCopypip install -r requirements.txt

Navigate to the inference folder:
bashCopycd inference

Run the inference:
bashCopypython run_inference.py

2. Using Python Virtual Environment

Create a virtual environment:
bashCopypython -m venv virtualenv

Activate the virtual environment:
bashCopy.\virtualenv\Scripts\activate

Follow steps 5-9 from the Conda instructions above.

Linux Instructions

1. Using Conda Environment

Ensure Conda is installed on your system.
Create a Conda Environment:
bashCopyconda create --name virtualenv

Activate the Conda environment:
bashCopyconda activate virtualenv

Install PyTorch:

For CPU:
bashCopyconda install pytorch torchvision cpuonly -c pytorch

For GPU:
bashCopyconda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

Clone the repository:
bashCopygit clone https://github.com/ByteMeEthos/CCTV_footage_detection.git

Navigate to the project directory:
bashCopycd CCTV_footage_detection

Install dependencies:
bashCopypip install -r requirements.txt

Navigate to the inference folder:
bashCopycd inference

Run the inference:
bashCopypython run_inference.py

2. Using Python Virtual Environment

Create a virtual environment:
bashCopypython3 -m venv virtualenv

Activate the virtual environment:
bashCopysource virtualenv/bin/activate

Follow steps 5-9 from the Conda instructions above.

Mac Instructions

1. Using Conda Environment

Ensure Conda is installed on your system.
Create a Conda Environment:
bashCopyconda create --name virtualenv

Activate the Conda environment:
bashCopyconda activate virtualenv

Install PyTorch:

For CPU:
bashCopyconda install pytorch torchvision -c pytorch

For GPU (Mac with M1 chip):
bashCopyconda install pytorch torchvision -c pytorch-nightly

Clone the repository:
bashCopygit clone https://github.com/ByteMeEthos/CCTV_footage_detection.git

Navigate to the project directory:
bashCopycd CCTV_footage_detection

Install dependencies:
bashCopypip install -r requirements.txt

Navigate to the inference folder:
bashCopycd inference

Run the inference:
bashCopypython run_inference.py

2. Using Python Virtual Environment

Create a virtual environment:
bashCopypython3 -m venv virtualenv

Activate the virtual environment:
bashCopysource virtualenv/bin/activate

Follow steps 5-9 from the Conda instructions above.

Viewing Results
After running the inference, you can find the model results in the data -> output directory.
