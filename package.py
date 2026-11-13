import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = ('module.yaml', 'compose.yaml', 'Dockerfile', '.dockerignore', 'requirements.txt',
         'server.py', 'model_files.py', 'skill_t2i.py', 'index.html', 'lora_library.json', 'README.md', 'package.py')


if __name__ == '__main__':
    version = next(line.split(':', 1)[1].strip() for line in (ROOT / 'module.yaml').read_text().splitlines()
                   if line.strip().startswith('version:'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', nargs='?', type=Path,
                        default=ROOT / 'dist' / f'skill-text-to-image-{version}.zip')
    destination = parser.parse_args().destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, name)
    print(destination)
