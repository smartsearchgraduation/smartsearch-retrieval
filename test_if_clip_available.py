# to see available clip models
import clip

print("CLIP available models:", clip.available_models())

# if you see errors, while in the virtual environment, try installing clip with these commands:

"venv/Scripts/python.exe -m pip uninstall clip -y"
"venv/Scripts/python.exe -m pip install git+https://github.com/openai/CLIP.git"

# then try again.
