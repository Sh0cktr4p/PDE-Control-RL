from setuptools import setup

setup(name='gym_phiflow',
      version='1.0.0',
      description='Controlling PDEs with Reinforcement Learning'
      install_requires=[
            'stable-baselines3', 
            'phiflow', 
            'pickle', 
            'matplotlib',
            'pandas'],
      author='Felix Trost',
      packages=['gym_phiflow']
)