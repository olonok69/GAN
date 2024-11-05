"""Cloud ML Engine package configuration."""
from setuptools import setup, find_packages

setup(name='Gan Model',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      author="juan Huertas",
      author_email="olonok@gmail.com",
      description='Gan model',
      install_requires=[
          'cherrypy==18.6.0',
          'flask==2.3.2',
          'graphviz==2.38',
          'keras==2.13.1',
          'matplotlib==3.1.3',
          'numpy==1.22.0',
          'pandas==1.0.3',
          'Pillow>=8.1.1',
          'pydot==1.4.1',
          'pyyaml==5.4',
          'scikit-learn==0.22.1',
          'seaborn==0.10.0',
          'tensorflow==2.11.1',
          'paste==3.2.2'
      ],
      python_requires='>=3.7',
      zip_safe=False)
