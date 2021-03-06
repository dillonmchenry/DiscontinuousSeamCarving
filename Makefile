.PHONY: test test_big debug

test: disc_video_carving.py
	python3 disc_video_carving.py --video slo_mo_guys_arrow.m4v --width 600 --height 480 --out bowdun.mp4

test_big: disc_video_carving.py
	python3 disc_video_carving.py --video bowser_dunk.mp4 --width 1920 --height 1080 --out bowdun.mp4

debug: disc_video_carving.py
	python3 -m pdb disc_video_carving.py --video bowser_dunk.mp4 --width 1920 --height 1080 --out bowdun.mp4