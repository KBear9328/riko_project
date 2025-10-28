"""
This script downloads and converts YouTube and YouTube Music playlists to mp3 files.
"""
import os
import re
from pytube import Playlist, YouTube

def download_playlist(playlist_url, output_dir, progress_queue=None):
    """
    Downloads a YouTube playlist and converts the videos to mp3 files.

    Args:
        playlist_url (str): The URL of the YouTube or YouTube Music playlist.
        output_dir (str): The directory to save the mp3 files to.
        progress_queue (queue.Queue, optional): A queue to report progress. Defaults to None.
    """
    def log(message):
        if progress_queue:
            progress_queue.put(message)
        else:
            print(message)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log(f"Created output directory: {output_dir}")

    try:
        playlist = Playlist(playlist_url)
        # This is a workaround for a pytube bug
        playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")
        log(f'Downloading playlist: {playlist.title}')
    except Exception as e:
        log(f"Error: Could not retrieve playlist. Please check the URL. Details: {e}")
        return

    for video_url in playlist.video_urls:
        try:
            yt = YouTube(video_url)
            log(f'Downloading: {yt.title}')

            audio_stream = yt.streams.filter(only_audio=True).first()
            if not audio_stream:
                log(f"No audio stream found for {yt.title}")
                continue

            # Download the audio stream
            audio_stream.download(output_path=output_dir)

            log(f'Successfully downloaded {yt.title}.')

        except Exception as e:
            log(f"An error occurred with {video_url}: {e}")

    log("--- Download complete! ---")
