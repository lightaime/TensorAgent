mode con: cols=80 lines=100


if NOT "%ComputerName%" == "PC-KW-60002" (
	set CUDA_VISIBLE_DEVICES=0 & activate tensorflow & python run_ddpg.py
) else (
	set CUDA_VISIBLE_DEVICES=0 & activate deep-fpv-racer & python run_ddpg.py & set /p temp="Hit enter to exit"
)