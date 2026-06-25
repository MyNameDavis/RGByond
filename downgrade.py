import subprocess
import os

def convert_video(input_file, output_file, width, height, fps=24):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    # Preserve aspect ratio and pad to exact dimensions
    vf_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )

    command = [
        'ffmpeg',
        '-i', input_file,
        '-vf', vf_filter,
        '-r', str(fps),
        '-c:v', 'libx264',
        '-crf', '20',
        '-preset', 'medium',
        '-c:a', 'copy',
        output_file
    ]

    try:
        print(f"Converting {input_file} to {width}x{height} at {fps}fps...")
        subprocess.run(command, check=True)
        print(f"Success! Saved as {output_file}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    input_vid = "cat_dog_480x360_30fps.mp4"

    # Example: same 4:3 ratio as 320x240
    output_width = 480
    output_height = 360
    fps = 30

    output_vid = f"roxy_{output_width}x{output_height}_{fps}fps.mp4"

    convert_video(
        input_vid,
        output_vid,
        output_width,
        output_height,
        fps
    )