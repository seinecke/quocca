from setuptools import setup, find_packages


setup(
    name='quocca',
    author='S. Einecke, T. Hoinka, H. Nawrath',
    author_email='sabrina.einecke@adelaide.edu.au, tobias.hoinka@tu-dortmund.de, helena.nawrath@tu-dortmund.de',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-image',
        'scikit-learn',
        'ruamel.yaml',
        'imageio',
        'scipy',
        'numpy',
        'astropy>=2',
    ],
    package_data={'quocca': ['resources/hipparcos.fits.gz',
                             'resources/catalogs.yaml',
                             'resources/cameras.yaml',
                             'resources/cta_mask.png']},
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
