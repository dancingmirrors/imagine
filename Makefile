PREFIX ?= /usr/local
BINDIR = $(PREFIX)/bin
SHAREDIR = $(PREFIX)/share/imagine
DESKTOPDIR = $(PREFIX)/share/applications

PYTHON = python3
VENV_DIR = $(SHAREDIR)/venv

.PHONY: all install uninstall clean help

all: help

help:
	@echo "Usage:"
	@echo "  make install      - Install to $(PREFIX)"
	@echo "  make uninstall    - Remove from $(PREFIX)"
	@echo "  make clean        - Clean the virtual environment"

install:
	@mkdir -p $(SHAREDIR)
	@mkdir -p $(BINDIR)
	@mkdir -p $(DESKTOPDIR)

	@install -m 755 imagine.py $(SHAREDIR)/imagine.py
	@install -m 644 requirements.txt $(SHAREDIR)/requirements.txt
	@install -m 644 README.md $(SHAREDIR)/README.md
	@install -m 644 COPYING $(SHAREDIR)/COPYING

	@if [ -d "$(VENV_DIR)" ] && [ -n "$(VENV_DIR)" ] && echo "$(VENV_DIR)" | grep -q "share/imagine/venv"; then \
		rm -rf $(VENV_DIR); \
	fi
	@$(PYTHON) -m venv $(VENV_DIR)

	@$(VENV_DIR)/bin/pip install --upgrade pip
	@$(VENV_DIR)/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
	@$(VENV_DIR)/bin/pip uninstall -y optimum
	@install -m 755 imagine $(BINDIR)/imagine

	@sed 's|@BINDIR@|$(BINDIR)|g' imagine.desktop.in > $(DESKTOPDIR)/imagine.desktop
	@chmod 644 $(DESKTOPDIR)/imagine.desktop

uninstall:
	@rm -f $(BINDIR)/imagine
	@rm -f $(DESKTOPDIR)/imagine.desktop
	@rm -rf $(SHAREDIR)

clean:
	@rm -rf venv
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@find . -type f -name '*.pyo' -delete 2>/dev/null || true
