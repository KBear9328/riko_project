"""
This script downloads and converts YouTube and YouTube Music playlists to mp3 files.
"""
import os
import re
import argparse
from pytube import Playlist, YouTube
from moviepy.editor import *

def download_playlist(playlist_url, output_dir):
    """
    Downloads a YouTube playlist and converts the videos to mp3 files.

    Args:
        playlist_url (str): The URL of the YouTube or YouTube Music playlist.
        output_dir (str): The directory to save the mp3 files to.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    playlist = Playlist(playlist_url)

    # This is a workaround for a pytube bug that prevents iterating through video_urls
    playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")

    print(f'Downloading playlist: {playlist.title}')

    for video_url in playlist.video_urls:
        try:
            yt = YouTube(video_url)
            print(f'Downloading: {yt.title}')

            audio_stream = yt.streams.filter(only_audio=True).first()
            if not audio_stream:
                print(f"No audio stream found for {yt.title}")
                continue

            # Download the audio stream
            output_file = audio_stream.download(output_path=output_dir)

            # Convert to mp3
            base, ext = os.path.splitext(output_file)
            new_file = base + '.mp3'

            # Check if the file already exists
            if os.path.exists(new_file):
                print(f'{yt.title} already exists. Skipping.')
                os.remove(output_file) # remove the original download
                continue

            audio_clip = AudioFileClip(output_file)
            audio_clip.write_audiofile(new_file)
            audio_clip.close()

            # Remove the original downloaded file
            os.remove(output_file)

            print(f'Successfully converted {yt.title} to mp3.')

        except Exception as e:
            print(f"An error occurred while downloading {video_url}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert a YouTube playlist to mp3.")
    parser.add_argument("playlist_url", help="The URL of the YouTube or YouTube Music playlist.")
    parser.add_argument("output_dir", help="The directory to save the mp3 files to.")
    args = parser.parse_args()

    download_playlist(args.playlist_url, args.output_dir)
