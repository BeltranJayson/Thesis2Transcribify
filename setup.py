import sys
import os
from cx_Freeze import setup, Executable

# Include any additional folders and files, like icons or resources, in this list
include_files = [
    "icon.ico",
    "ffmpeg", 
    "model", 
    "resemblyzer", 
    "uploads",
    "ui_main.py",
    "main_function.py"
]

# Define the target executable
target = Executable(
    script="main.py",
    base="Win32GUI" if sys.platform == "win32" else None,
    icon="icon.ico",
)

# Setup configuration
setup(
    name="Transcription",
    version="1.0",
    description="Transcription and Diarization System",
    author="Jayson Beltran",
    options={
        'build_exe': {
            'include_files': include_files,
            'packages': ["os", "sys"],
        },
    },
    executables=[target]
)
