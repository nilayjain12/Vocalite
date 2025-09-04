# 🎤 Vocalite - Real-Time Meeting Transcription Tool

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Windows](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)](https://microsoft.com/windows)

**Vocalite** is a powerful desktop application that provides real-time transcription of meetings, interviews, and conversations. It captures audio from both your microphone and system audio (like Zoom, Teams, or any application) and displays live transcriptions with speaker identification and timestamps.

## 🎯 Problem It Solves

- **Missing Important Details**: Never miss crucial information during interviews or meetings again
- **Note-Taking Distraction**: Focus on the conversation instead of frantically taking notes
- **Post-Meeting Review**: Analyze your performance in interviews or review meeting outcomes
- **Accessibility**: Provides real-time text for better comprehension and accessibility
- **Documentation**: Automatically creates professional transcripts with timestamps

## ✨ Key Features

### 🎙️ Real-Time Transcription
- **Live Display**: See transcriptions appear instantly as people speak
- **Dual Audio Capture**: Records both microphone input and system audio simultaneously
- **Speaker Identification**: Distinguishes between "You (Mic)" and "Other (System)" speakers
- **Professional Timestamps**: Every line includes precise timing information

### 🤖 AI-Powered Analysis
- **Interview Analysis**: Get detailed feedback on interview performance with strengths, weaknesses, and improvement recommendations
- **Meeting Summaries**: Generate structured meeting minutes with action items and key decisions
- **Smart Summarization**: Uses Groq's advanced language models for intelligent content analysis

### 💻 User-Friendly Interface
- **Simple GUI**: Clean, intuitive interface built with Tkinter
- **One-Click Recording**: Start and stop recording with simple buttons
- **Live Preview**: Real-time transcript display during recording
- **Export Options**: Save transcripts and summaries in various formats

### 🔧 Technical Excellence
- **High Accuracy**: Uses Groq's Whisper Large V3 model for superior transcription quality
- **Cross-Platform Audio**: Supports Windows system audio capture via PyAudioWPatch
- **Standalone Executable**: No Python installation required - includes all dependencies
- **FFmpeg Integration**: Professional audio processing capabilities

## 🚀 Quick Start

### Prerequisites
- **Windows 10/11** (system audio capture is Windows-specific)
- **Groq API Key** (free tier available at [console.groq.com](https://console.groq.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nilayjain12/Vocalite.git
   cd Vocalite
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python audiotranscriber_v5_working.py
   ```

### First-Time Setup

1. **Configure API Key**: Go to `Settings > Set Groq API Key` and enter your Groq API key
2. **Test Audio**: Ensure your microphone is working and system audio is available
3. **Start Recording**: Click "🎬 Start Recording" to begin real-time transcription

## 📖 Usage Guide

### Basic Recording
1. **Start Recording**: Click the "🎬 Start Recording" button
2. **Live Transcription**: Watch the transcript appear in real-time
3. **Stop & Save**: Click "⏹️ Stop & Save" to finalize and save the transcript

### Generating Summaries
1. **Select Summary Type**: Choose between "Summarize Interview" or "Summarize Meeting"
2. **Generate Summary**: Click "Summarize" to create an AI-powered analysis
3. **Save Summary**: Use "Save Summary As..." to export the summary

### Working with Existing Transcripts
1. **Upload Transcript**: Use "Upload Transcript..." to load existing text files
2. **Preview & Edit**: View and modify content in the Live Transcript area
3. **Generate Analysis**: Create summaries from uploaded transcripts

## 🛠️ Building from Source

### Development Setup
1. **Clone and navigate to project**
   ```bash
   git clone https://github.com/nilayjain12/Vocalite.git
   cd Vocalite
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r "requirements.txt"
   ```

### Building Executable
```bash
pyinstaller Vocalite.spec
```

The executable will be created in the `dist/Vocalite/` directory.

## 📝 Sample Output

### Interview Transcript
```
[00:00:00] Other (System): Thank you for joining us today. Can you tell me about yourself?
[00:00:05] You (Mic): I'm a software engineer with 5 years of experience in Python and machine learning.
[00:00:15] Other (System): That's great. What interests you about this role?
[00:00:20] You (Mic): I'm excited about the opportunity to work on AI-powered applications...
```

### Generated Interview Analysis
```
## Interview Performance Analysis

### Executive Summary
Overall performance was strong with good technical communication and clear examples.

### Strengths
- Provided specific examples from past projects
- Asked thoughtful questions about the role
- Demonstrated strong technical knowledge

### Areas for Improvement
- Could elaborate more on leadership experience
- Consider asking more about team dynamics

### Recommendations
1. Prepare more detailed examples of project leadership
2. Research the company's recent developments
3. Practice explaining complex technical concepts simply
```

## 🤝 Contributing

Contributions are welcomed! Feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Cloud storage integration
- [ ] Mobile app companion
- [ ] Advanced speaker diarization
- [ ] Custom vocabulary support
- [ ] Integration with popular meeting platforms

---

**Made with ❤️ for better meeting experiences**

*Vocalite - Never miss a word again.*
