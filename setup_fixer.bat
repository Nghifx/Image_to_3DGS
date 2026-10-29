@echo off
rem GSplat Studio - moi truong cho NVIDIA Fixer (tools\fixer_venv). Chay 1 lan sau setup.bat, can Internet (~4 GB).
rem Fixer = tools\Fixer (code: git clone nv-tlabs/Fixer, models: Hugging Face nvidia/Fixer - xem README; trong goi
rem zip noi bo thi co san). torch cu128; cosmos-predict2 cai --no-deps vi
rem transformer-engine / megatron-core khong build duoc tren Windows (tools\Fixer\win_shims thay the).
cd /d "%~dp0"
if not exist tools\fixer_venv\Scripts\python.exe py -3.13 -m venv tools\fixer_venv || python -m venv tools\fixer_venv
set FPY=tools\fixer_venv\Scripts\python.exe
%FPY% -m pip install --upgrade pip
%FPY% -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
%FPY% -m pip install -r tools\Fixer\requirements_fixer_win.txt
%FPY% -m pip install cosmos-predict2==1.0.9 --no-deps
%FPY% -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pause
