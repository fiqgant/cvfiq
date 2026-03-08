# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.3.x | ✅ Active |
| 0.2.x | ❌ No longer supported |
| < 0.2 | ❌ No longer supported |

## Reporting a Vulnerability

If you discover a security vulnerability, **please do not open a public GitHub issue**.

Instead, report it privately by emailing:

**fiqgant@gmail.com**

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive a response within **72 hours**. If the issue is confirmed, a patch will be released as soon as possible.

## Scope

This library processes local camera/video input and does not handle network requests, user authentication, or sensitive data by default. Security concerns most relevant to this project:

- Malicious model files (`.task`, `.tflite`, `.onnx`) — only load models from trusted sources
- Untrusted video input that could exploit OpenCV parsing vulnerabilities
- Serial port communication (`SerialModule`) — ensure trusted devices only
