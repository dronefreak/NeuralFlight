# Security Policy

## Supported Versions

We take security seriously for NeuralFlight. The following versions are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in NeuralFlight, please report it privately to help us address it before public disclosure.

### How to Report

1. **Email**: Send details to `kumaar324@gmail.com`
2. **Subject**: Include "SECURITY" in the subject line
3. **Details**: Provide the following information:
   - Type of vulnerability
   - Affected component(s)
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Updates**: Regular status updates every 5 business days
- **Timeline**: We aim to address critical vulnerabilities within 7 days
- **Credit**: We will acknowledge your contribution in the fix announcement (unless you prefer to remain anonymous)

## Security Considerations

### General Safety

NeuralFlight is designed for **research and educational purposes**. When using this software:

- ✅ **DO**: Use in controlled environments (simulation, testing areas)
- ✅ **DO**: Test thoroughly before any real-world applications
- ✅ **DO**: Follow local regulations for drone operation
- ❌ **DON'T**: Use in safety-critical applications without extensive validation
- ❌ **DON'T**: Rely solely on BCI control for autonomous systems
- ❌ **DON'T**: Operate real drones near people or property without proper safeguards

### Known Limitations

1. **EEG Classification**: 73% accuracy means ~27% error rate - not suitable for critical applications
2. **Real-time Performance**: Latency exists between thought and action
3. **Subject Variability**: Performance varies significantly between individuals
4. **Environmental Factors**: EEG signals are affected by electrical interference, muscle artifacts, and environmental noise

### Hardware Safety

If adapting for real drones:

- Implement multiple redundant safety systems
- Always include manual override capability
- Use geofencing and altitude limits
- Test extensively in simulation before hardware deployment
- Follow manufacturer safety guidelines

## Dependencies

We regularly monitor our dependencies for known vulnerabilities:

- **PyTorch**: We track CVEs and update when security patches are released
- **OpenCV**: Updated regularly for security fixes
- **Mediapipe**: Google maintains this library with regular security updates
- **MNE-Python**: Community-maintained with security considerations

You can check for vulnerable dependencies using:

```bash
pip install safety
safety check -r requirements.txt
```

## Data Privacy

### EEG Data Handling

NeuralFlight processes EEG data locally:

- ✅ No data is transmitted to external servers
- ✅ All processing happens on your local machine
- ✅ You control data storage and deletion
- ✅ PhysioNet dataset usage follows their terms of service

### User Privacy

- We do not collect any personal information
- Webcam data (for gesture control) is processed locally only
- No telemetry or analytics are collected
- No third-party tracking

## Best Practices for Contributors

If you're contributing code:

1. **Input Validation**: Always validate user inputs
2. **Error Handling**: Handle exceptions gracefully
3. **Dependencies**: Only add well-maintained, security-vetted packages
4. **Sensitive Data**: Never commit API keys, credentials, or personal data
5. **Code Review**: All PRs undergo security review before merging

## Security Updates

We will announce security updates through:

- GitHub Security Advisories
- Release notes
- README notifications

Subscribe to repository notifications to stay informed.

## Scope

This security policy covers:

- ✅ The core NeuralFlight codebase
- ✅ Official demo applications
- ✅ Documentation and examples
- ❌ Third-party integrations (report to those projects)
- ❌ User-modified versions (your responsibility)

## Questions?

For general security questions (not vulnerability reports), open a GitHub Discussion or contact us at `kumaar324@gmail.com`.

---

**Remember**: NeuralFlight is an experimental research platform. Always prioritize safety when working with autonomous systems!
