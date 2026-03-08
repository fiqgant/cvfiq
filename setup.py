from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cvfiq',
    packages=find_packages(),
    version='0.4.2',
    license='MIT',
    description='Computer Vision Helping Library/Functions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Fiqgant',
    author_email='fiqgant@gmail.com',
    url='https://github.com/fiqgant/cvfiq.git',
    keywords=['ComputerVision', 'HandTracking', 'FaceTracking', 'PoseEstimation'],
    install_requires=[
        # OpenCV — min 4.7.0 required for cv2.aruco.ArucoDetector
        'opencv-python>=4.7.0,<=4.13.0.92',
        # MediaPipe — min 0.10.0 required for Tasks API (GestureRecognizer, FaceLandmarker, ObjectDetector)
        'mediapipe>=0.10.0,<=0.10.32',
        # NumPy — min 1.21.0, supports numpy 2.x
        'numpy>=1.21.0,<=2.4.2',
        # PySerial — required for SerialModule (Arduino communication)
        'pyserial>=3.0,<=3.5',
    ],
    extras_require={
        # Optional: required for ClassificationModule (choose one)
        'keras': ['keras>=2.12.0'],
        'tensorflow': ['tensorflow>=2.12.0'],
    },
    python_requires='>=3.8,<3.13',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
