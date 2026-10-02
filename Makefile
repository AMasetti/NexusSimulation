# NexusSimulation — Makefile (Optimus targets; spot_micro has none yet)
#
#   make install         pip install -r requirements.txt + git hook
#   make run             full-body Optimus sim in the MuJoCo viewer
#   make cpg-train       train CPG+PPO residual policy (headless)
#   make cpg-eval        run the newest CPG+PPO checkpoint in the viewer
#
# The viewer needs mjpython on macOS; training is headless and uses plain python.
# Override either:  make run MJPYTHON=mjpython   make cpg-train PYTHON=python

.PHONY: help install hooks run rl-eval cpg cpg-train cpg-eval monitor

ROOT     := $(shell pwd)
RL_DIR   := $(ROOT)/optimus/rl
MJPYTHON ?= $(HOME)/.pyenv/versions/3.10.0/envs/robotics/bin/mjpython
PYTHON   ?= python3

# Newest run that has a best_model.zip. Override: make cpg-eval MODEL=checkpoints/<run>/best_model.zip
MODEL ?= $(shell cd $(RL_DIR) && for d in $$(ls -td checkpoints/*_run_* 2>/dev/null); do [ -f "$$d/best_model.zip" ] && echo "$$d/best_model.zip" && break; done)

help:
	@echo ""
	@echo "  NexusSimulation (MuJoCo)"
	@echo "  ════════════════════════════════════════════════"
	@echo "  make install       pip install -r requirements.txt + git hook"
	@echo "  make hooks         enable the commit-msg hook"
	@echo ""
	@echo "  Optimus"
	@echo "    make run         full-body sim in the viewer"
	@echo "    make rl-eval     best plain-PPO policy in the viewer"
	@echo "    make cpg         CPG gait only (no RL) in the viewer"
	@echo "    make cpg-train   train CPG+PPO (headless)"
	@echo "    make cpg-eval    newest CPG+PPO checkpoint in the viewer  [MODEL=...]"
	@echo "    make monitor     poll the latest training run, notify via Telegram"
	@echo ""

install: hooks
	$(PYTHON) -m pip install -r $(ROOT)/requirements.txt

hooks:
	@git -C $(ROOT) config core.hooksPath .githooks && echo "[hooks] NexusSimulation → .githooks"

run:
	$(MJPYTHON) $(ROOT)/optimus/urdf/full/Launch_mujuco.py

rl-eval:
	cd $(RL_DIR) && $(MJPYTHON) train.py --eval checkpoints/best/best_model.zip

cpg:
	cd $(RL_DIR) && $(MJPYTHON) train_cpg.py --cpg-only

cpg-train:
	cd $(RL_DIR) && $(PYTHON) train_cpg.py

cpg-eval:
	@if [ -z "$(MODEL)" ]; then echo "No checkpoint with best_model.zip found. Pass MODEL=..."; exit 1; fi
	@echo "[sim] Evaluating $(MODEL)"
	cd $(RL_DIR) && $(MJPYTHON) train_cpg.py --eval $(MODEL)

monitor:
	cd $(RL_DIR) && $(PYTHON) monitor.py
