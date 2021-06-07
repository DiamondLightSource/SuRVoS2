set root=C:\ProgramData\Miniconda3

call %root%\Scripts\activate.bat %root%

call conda activate survos2_env

call cd "C:\Program Files\SuRVoS\SuRVoS2-master"

call python -m survos2.frontend.runner


