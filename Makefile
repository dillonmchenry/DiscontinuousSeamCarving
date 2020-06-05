.PHONY: test debug

test: disc_video_carving.py
	python3 disc_video_carving.py --video bowser_dunk.mp4 --width 1920 --height 1080 --out bowdun.mp4

debug: disc_video_carving.py
	python3 -m pdb disc_video_carving.py --video bowser_dunk.mp4 --width 1920 --height 1080 --out bowdun.mp4