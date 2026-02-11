# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import sys
from pathlib import Path

# Paths
backend_dir = Path(".").absolute()
project_dir = backend_dir.parent
dist_dir = project_dir / "dist"

# Helper to collect all package data
from PyInstaller.utils.hooks import collect_all

# Collect everything from torch, torchvision, basicsr and realesrgan
# This ensures CUDA binaries are included for GPU acceleration
torch_datas, torch_binaries, torch_hidden = collect_all('torch')
torchvision_datas, torchvision_binaries, torchvision_hidden = collect_all('torchvision')
basicsr_datas, basicsr_binaries, basicsr_hidden = collect_all('basicsr')
realesrgan_datas, realesrgan_binaries, realesrgan_hidden = collect_all('realesrgan')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=torch_binaries + torchvision_binaries + basicsr_binaries + realesrgan_binaries,
    datas=[
        (str(dist_dir), 'dist'),             # Include Frontend Build to internal 'dist' folder
        ('queue_manager.py', '.'),           # Include Python modules
        ('upscaler.py', '.'),
    ] + torch_datas + torchvision_datas + basicsr_datas + realesrgan_datas,    # Add collected datas
    hiddenimports=[
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'engineio.async_drivers.aiohttp',
        'torch',
        'torchvision',
        'basicsr',
        'realesrgan',
        'cv2',
        'numpy',
        'numpy.core',
        'numpy.lib',
        'numpy.random',
        'numpy.testing',
        'numpy.f2py',
        'numpy.distutils',
    ] + torch_hidden + torchvision_hidden + basicsr_hidden + realesrgan_hidden,  # Add collected hidden imports
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='UpscalerAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False to hide terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,     # You can add an icon file here (e.g., 'app.ico')
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='UpscalerAI',
)
