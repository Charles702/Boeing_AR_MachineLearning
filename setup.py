from setuptools import setup, find_packages

setup(name='commontools_ar',
      version='0.1',
      description='Data tools used for Bowing anchoring project',
      url='https://github.com/Charles702/Boeing_AR_tools',
      author='Chen Zhu',
      author_email='chen_zhu@sfu.ca',
      packages=find_packages(),
      package_data={'commontools_ar': ['resources/*']},
      include_package_data=True,
      zip_safe=False,
      install_requires=['matplotlib'],
      entry_points={ 
        'console_scripts': ['my-command=commontools_ar.test:main'] }
      )