# Barcodes and 2D codes

Some collections print a barcode, a QR code or a Data Matrix on their tags, next to the
catalog number or instead of it. TagSort decodes these codes before it reads any text. A
code is not read but decoded: its own checksums make the text certain, so a code that
fits the profile is accepted without OCR and without a vision API.

```python
reader = Reader(profile)  # decodes codes whenever the extra is installed
reader = Reader(profile, codes=False)  # text only
```

Decoding needs the `codes` extra, which installs
[zxing-cpp](https://github.com/zxing-cpp/zxing-cpp) (Apache 2.0):

```bash
pip install "tagsort[codes]"
```

The Docker images include it, so the server decodes codes with no setting. With
`codes=True` the reader refuses to start without the extra; a `CodeDecoder` of your own
(any object with `decode(image) -> list[DecodedCode]`) replaces zxing-cpp.

## Which codes

Every format zxing-cpp knows, among them 1D barcodes (Code 128, Code 39, Code 93, Codabar,
EAN, UPC, ITF), QR, Micro QR, Data Matrix, Aztec and PDF417, in any orientation.

## What comes back

A decoded code becomes a tag when its content fits a kind of tag in the profile:

| Content | Tag |
| --- | --- |
| `GJ02948` | `GJ02948` |
| `GJ 02948` | `GJ02948` (spaces removed) |
| `https://collections.example.org/specimen/GJ02948` | `GJ02948` (the part of a URL that fits) |
| `gj02948` | `GJ02948` (a part in lower case is also tried in upper case) |

Such a tag has `source: "barcode"` (`result.v1.json`), confidence 1 and status
`accepted`. Its `polygon` and `angle` are those of the code. When two parts of one code fit
the profile (a URL holding two numbers), the first is the tag and the others are its
candidates, with status `review`: a person chooses.

The text printed under a barcode usually says the same thing. It is not returned a second
time: a line read by OCR that equals a decoded tag is dropped, with a local backend and
with a vision API alike.

A code whose content fits no kind of tag is not a tag. With
[open reading](open-reading.md), it comes back in `other_texts`, with confidence 1 and a
guess of what it is, so an application can show it to a person.

## Cost

Decoding takes about 20 ms for a 2048-pixel photo on a laptop CPU (`timings_ms.codes` with
a local backend). With a vision API it costs nothing more: the codes are decoded locally,
before the photo is sent.
