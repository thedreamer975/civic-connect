import subprocess
import sys

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

subprocess.check_call([sys.executable, '-m', 'pip', 'install', *requirements])
print('All dependencies installed.')
