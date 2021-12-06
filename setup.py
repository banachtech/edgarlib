from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]

install_requires = [
  'pandas',
  'pandas_datareader',
  'yahooquery',
  'yfinance',
  'wheel',
  'scikit-learn'
]

setup(
  name='edgar',
  version='0.0.1',
  description='E',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='Louis Hoo',
  author_email='louis.h5227@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='edgar', 
  packages=find_packages(),
  install_requires=install_requires
)
