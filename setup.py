import setuptools

setuptools.setup(
        name = 'mopu',
        version = '0.1.0',
        packages = setuptools.find_packages(),
        install_requires = ['numpy', 'torch'],
        entry_points = {
            'console_scripts':[
                'tunimi = tunimi.main:main',
                'wannimi = tunimi.wannimi:main',
                'train-toki = toki.train:main',
                'topp-toki = toki.generate:main',
                ]},)
