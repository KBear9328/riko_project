# YouTube Playlist Audio Downloader

This is a simple web-based tool to download the audio from YouTube playlists. It provides a user-friendly interface to start the download process and view the progress in real-time.

## Features

- **Web-Based UI:** Easy-to-use web interface for downloading playlists.
- **Real-Time Progress:** View the download progress directly on the webpage.
- **High-Quality Audio:** Downloads the best available audio stream in its original format (e.g., `.m4a`, `.webm`).

## How to Use

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Web Server:**
   ```bash
   python web_server.py
   ```

3. **Open the Web Interface:**
   Open your web browser and go to `http://127.0.0.1:8000`.

4. **Start Downloading:**
   - Enter the URL of the YouTube playlist.
   - Enter the directory where you want to save the audio files.
   - Click "Download".

## How It Works

This tool is built with Python and uses the following libraries:

- **Flask:** To create the web server and user interface.
- **Pytube:** To interact with YouTube and download the audio streams.

When you start a download, the web server runs the downloader in a background thread to keep the UI responsive. The progress is sent back to the web page and displayed in real-time.
