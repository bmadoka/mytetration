"""The original files sit at the fork's root (make_reference.original_root()), byte-for-byte as in DMTPARK/mytetration
commit 21b26204 (after undoing git's line-ending conversion: a Windows checkout with core.autocrlf=true writes CRLF
into the text files). README.md is the one exception: the fork may add its own section on top, and the original text
must stay as its tail."""
import hashlib

import make_reference

COMMIT = "21b26204e874b2d5b8e6237f1941b5f0b9fd3418"
# path: (bytes, sha256) of the git blobs of that commit
FILES = {
    "Power Tower Fractal (static)/PTF_static_16by9_H.py":
        (1918, "f957ccf59647c158a6cbefe45e5711d0ae4c0659cec45e4f92c96c68870ce755"),
    "Power Tower Fractal (static)/PTF_static_16by9_R.py":
        (1999, "ed845abfcce488b40f4f7a82f75cb0119572b466fb6262e25c110428528f5e49"),
    "Power Tower Fractal (static)/PTF_static_1by1_H.py":
        (1791, "2969f16cfdbb32e6f2264bf60a03fbec668c14dcb870ccd39b56476542faf7d7"),
    "Power Tower Fractal (static)/PTF_static_1by1_R.py":
        (1873, "984e0356557e2a45283243e490914ee0852d5893adf15b0dc10d4fe67e2f238b"),
    "Power Tower Fractal (static)/PTF_static_4by5_H.py":
        (1915, "137d739c7de6c77bb98e92dfb49e0d76e4849294b83b63cce213474e6dba617a"),
    "Power Tower Fractal (static)/PTF_static_4by5_R.py":
        (1996, "211e60f0f4a8beeaa74182e3e031ebd305dfc12a271e442c309851c9c16956f9"),
    "Power Tower Fractal (static)/PTF_static_9by16_H.py":
        (1918, "4ab33cf808f19a7adc83486ef60207ca6d39f9de9e14f16f8ef4562a63c7878e"),
    "Power Tower Fractal (static)/PTF_static_9by16_R.py":
        (1999, "ee213713935b8dfa9e14dde07a99c4cca4307465db6720d0157b14948ef8ab51"),
    "Power Tower Fractal (static)/sample/mytetration_x_0_y_0_eps_5.png":
        (54138, "bea130d4cb01cefd78f70cd576c30d315d8577c41d2b86af5f4755beec3cfd4d"),
    "Power Tower Fractal (zoom-in)/PTF_zoom.py":
        (2655, "26b66c2203e1e5e8853e5905edab015d91dc2596a93fe121dcc26b5ca13935c7"),
    "README.md": (16119, "c80da6f3c09673e82461edc9508c5aa3eb86b4adf790299454ab7a8045173a22"),
    "go further/PTF_sqrt2.py": (1534, "f22d4a1b3f70437d98afcd36548de85933ac4250107c07ac7d86e0cbef83d1f7"),
    "go further/Sample Image for 'PTF_sqrt2.py'.png":
        (122437, "5815e06e370a40074a32bf289ffd5f1e4769379e895268c3d842cffad1b37147"),
}


def original_bytes(path):
    b = (make_reference.original_root() / path).read_bytes()
    b = b if path.endswith(".png") else b.replace(b"\r\n", b"\n")
    if path == "README.md":  # the fork may add its own section on top; the original text must stay as the tail
        b = b[-FILES[path][0]:]
    return b


def test_original_files_at_the_fork_root():
    got = {p: (len(original_bytes(p)), hashlib.sha256(original_bytes(p)).hexdigest()) for p in FILES}
    bad = {p: (g[0], g[1][:12]) for p, g in got.items() if g != FILES[p]}
    assert not bad, f"not the files of commit {COMMIT}: {bad}"
