import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='cppn_pytorch',
    version='0.0.3',
    author='Jackson Dean',
    author_email='jackson@downbeat.games',
    description='CPPN implementation in PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jacksonsdean/cppn_pytorch',
    project_urls = {
        "Bug Tracker": "https://github.com/jacksonsdean/cppn_pytorch/issues"
    },
    license='MIT',
    packages=['cppn_pytorch'],
    install_requires=['torch', 'numpy', 'matplotlib', 'scipy', 'scikit-image', 'functorch', 'tqdm', 'torchvision', 'piq', 'imageio','scikits-bootstrap', 'networkx'],
)