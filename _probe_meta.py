from PIL import Image
import piexif, re

path = '/Users/corey/Desktop/hpmTestFiles/ _JPEG/BirdLady_24Bit_g2p0004.tif'
img = Image.open(path)

# Try piexif directly on file
try:
    exif = piexif.load(path)
    print('piexif direct load worked')
    for ifd in exif:
        for k, v in exif[ifd].items():
            val = v[:80] if isinstance(v, bytes) else str(v)[:80]
            print(f'  [{ifd}] {k}: {val}')
except Exception as e:
    print('piexif direct error:', e)

# XMP
xmp = img.info.get('xmp', b'')
if isinstance(xmp, bytes):
    xmp = xmp.decode(errors='replace')
print('\nLens from XMP:', re.findall(r'(?:LensModel|aux:Lens|exifEX:LensModel)[^>]*>([^<]+)', xmp))
print('Focal from XMP:', re.findall(r'FocalLength[^>]*>([^<]+)', xmp))
print('Aperture from XMP:', re.findall(r'FNumber[^>]*>([^<]+)', xmp))

# TIFF tags
tag_names = {271: 'Make', 272: 'Model', 305: 'Software', 34665: 'ExifIFD_offset'}
print('\nTIFF tags:')
for k, name in tag_names.items():
    v = img.tag_v2.get(k)
    if v:
        print(f'  {name}: {v}')
