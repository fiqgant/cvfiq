# Contributing to cvfiq

Thank you for your interest in contributing! Here's how to get started.

## Ways to Contribute

- Report bugs via [GitHub Issues](https://github.com/fiqgant/cvfiq/issues)
- Suggest new features or modules
- Fix bugs or improve existing modules
- Improve documentation or examples
- Add or fix tests in the `tests/` folder

## Getting Started

```bash
# Fork and clone the repo
git clone https://github.com/YOUR_USERNAME/cvfiq.git
cd cvfiq

# Create a virtual environment
conda create -n cv python=3.10
conda activate cv

# Install in editable mode
pip install -e .
pip install opencv-python mediapipe numpy pyserial
```

## Development Guidelines

### Code Style
- Follow existing code style in each module
- Keep functions focused and well-documented
- Use descriptive variable names

### Adding a New Module

1. Create `cvfiq/YourModule.py` following this structure:

```python
"""
YourModule — short description.
"""

import cv2
import mediapipe as mp


class YourDetector:
    def __init__(self, param=default):
        """
        :param param: description
        """
        pass

    def detect(self, img, draw=True):
        """
        :param img: Input BGR image
        :param draw: Draw results on image
        :return: results, image
        """
        pass


def main():
    cap = cv2.VideoCapture(0)
    detector = YourDetector()
    while True:
        success, img = cap.read()
        results, img = detector.detect(img)
        cv2.imshow("Test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

2. Export it in `cvfiq/__init__.py`
3. Add a test script in `tests/test_yourmodule.py`
4. Document it in `README.md`

### Adding Tests

Place test scripts in `tests/`. Camera-based tests should:
- Import from `..` (parent directory)
- Have a `main()` function
- Handle missing camera gracefully
- Use `q` key to quit

### Running Tests

```bash
cd tests
python run_all.py        # batch tests (no camera)
python test_yourmodule.py  # individual test
```

## Submitting a Pull Request

1. Create a new branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run the batch tests: `python tests/run_all.py`
4. Commit with a clear message
5. Push and open a Pull Request

### PR Checklist
- [ ] Code follows existing style
- [ ] Tests added/updated
- [ ] README updated if needed
- [ ] No breaking changes to existing APIs

## Reporting Bugs

Please include:
- Python version and OS
- cvfiq version (`pip show cvfiq`)
- OpenCV and MediaPipe versions
- Minimal code to reproduce the issue
- Full error traceback
