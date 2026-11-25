import sys, os

# adjust this path to match the SDK installation
sdk_base = "/usr/local/amba-cv-tools-cv7x.3.8.3.0.2204.ubuntu-22.04/cv72/tv2/release"
# find the python3.10 site-packages directory recursively
site_dirs = []
for root, dirs, files in os.walk(sdk_base):
    if root.endswith("site-packages"):
        site_dirs.append(root)
        break
if not site_dirs:
    print("cannot locate sdk site-packages")
    sys.exit(1)
site = site_dirs[0]
# write log to workspace root
logfile = open(os.path.join(os.getcwd(),"parser_inspect.log"),"w")
def log(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=logfile)

log("using sdk site-packages:", site)
sys.path.insert(0, site)

try:
    import frameworklibs.onnx as onnxmod
except Exception as e:
    log("failed to import frameworklibs.onnx", e)
    sys.exit(1)

log("module attributes:", [n for n in dir(onnxmod) if not n.startswith("__")])

# try to see if there's a parse or parser class
for name in dir(onnxmod):
    if "parse" in name.lower() or "parser" in name.lower():
        log("found", name)

# optionally, call a parse routine if present
if hasattr(onnxmod, "onnx_parser"):
    print("onnx_parser attr", onnxmod.onnx_parser)

# maybe the entrypoint is a function named parse or Parse
if hasattr(onnxmod, "parse"):
    print("calling parse function...")
    # sample invocation; adapt as needed
    # onnxmod.parse(input="ctdet_coco_dla_2x_conv_v2.onnx", output="model_ir")

log("script done")
logfile.close()
