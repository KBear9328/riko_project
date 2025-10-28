from flask import Flask, render_template, request, jsonify
import threading
import queue
from youtube_downloader import download_playlist

app = Flask(__name__)

# A queue to hold the progress messages
progress_queue = queue.Queue()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download', methods=['POST'])
def download():
    data = request.get_json()
    playlist_url = data.get('playlist_url')
    output_dir = data.get('output_dir')

    if not playlist_url or not output_dir:
        return jsonify({'error': 'Playlist URL and Output Directory are required.'}), 400

    # Clear the queue before starting a new download
    while not progress_queue.empty():
        progress_queue.get()

    # Run the download in a separate thread
    download_thread = threading.Thread(target=download_playlist, args=(playlist_url, output_dir, progress_queue))
    download_thread.start()

    return jsonify({'message': 'Download started!'})

@app.route('/progress')
def progress():
    messages = []
    while not progress_queue.empty():
        messages.append(progress_queue.get())
    return jsonify({'messages': messages})

if __name__ == '__main__':
    app.run(port=8000)
