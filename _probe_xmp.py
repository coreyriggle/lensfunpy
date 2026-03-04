from PIL import Image
import re

path = '/Users/corey/Desktop/hpmTestFiles/ _JPEG/BirdLady_24Bit_g2p0004.tif'
img = Image.open(path)
xmp = img.info.get('xmp', b'')
if isinstance(xmp, bytes):
    xmp = xmp.decode(errors='replace')

# Print all XMP lines containing lens-related keywords
keywords = ['lens', 'Lens', 'focal', 'Focal', 'camera', 'Camera', 'make', 'Make', 'model', 'Model']
for line in xmp.splitlines():
    if any(k in line for k in keywords):
        print(line.strip())
