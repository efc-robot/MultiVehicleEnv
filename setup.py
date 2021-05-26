from setuptools import setup, find_packages

setup(name='MultiVehicleEnv',
      version='0.0.1',
      description='Multi-Vehicle Environment',
      author='Jiantao Qiu',
      author_email='qjt15@mails.tsinghua.edu.cn',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False
      #install_requires=['gym', 'numpy-stl', 'numpy', 'six', 'pyglet']
)
