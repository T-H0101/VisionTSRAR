"""
VisionTSRAR 安装配置

用法:
    pip install -e .         # 开发模式安装
    pip install .            # 正式安装
"""

from setuptools import setup, find_packages

setup(
    name='visiontsrar',
    version='0.1.0',
    description='VisionTSRAR: Time Series Forecasting via Randomized Autoregressive Visual Generation',
    long_description=open('README.md', encoding='utf-8').read() if __import__('os').path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='VisionTSRAR Team',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.1.0',
        'torchvision>=0.16.0',
        'einops>=0.7.0',
        'Pillow>=9.0.0',
        'omegaconf>=2.3.0',
        'safetensors>=0.4.0',
        'huggingface_hub>=0.19.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'scipy>=1.11.0',
        'matplotlib>=3.7.0',
        'tqdm>=4.65.0',
    ],
    extras_require={
        'jupyter': ['jupyter>=1.0.0', 'ipython>=8.14.0'],
        'eval': ['gluonts>=0.14.0'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
