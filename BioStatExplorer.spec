# -*- mode: python ; coding: utf-8 -*-
# Builds the Mac app bundle and the single-file Windows executable from one
# recipe. The Windows job used to call PyInstaller directly with --onefile,
# which skipped the hiddenimports below: matplotlib picks its output writer
# from the file extension at save time, so SVG and PDF export were missing
# from every Windows build until this file was used for both platforms.
import sys

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=[('assets/icon.png', 'assets')],
    # figure exports load these backends by name at save time
    hiddenimports=['matplotlib.backends.backend_svg',
                   'matplotlib.backends.backend_pdf',
                   'matplotlib.backends.backend_ps',
                   'matplotlib.backends.backend_agg',
                   'statsmodels.stats.anova',
                   'statsmodels.stats.multitest'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # one Qt binding only (PySide6); skip big unused packages
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'tkinter', 'IPython', 'jupyter',
              'notebook', 'sphinx', 'pytest', 'pingouin', 'scikit_posthocs'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

if sys.platform == 'darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='BioStatExplorer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='BioStatExplorer',
    )
    app = BUNDLE(
        coll,
        name='BioStatExplorer.app',
        icon='assets/icon.icns',
        bundle_identifier='uk.ac.earlham.biostatexplorer',
        info_plist={'CFBundleShortVersionString': '2.0.0',
                    'NSHighResolutionCapable': True},
    )
else:
    # Single-file Windows executable: everything is bundled into the EXE
    # itself, so there is one file to download and double-click.
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='BioStatExplorer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        icon='assets/icon.ico',
    )
