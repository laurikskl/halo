from PIL import Image
import os

def resize_images(directory, output_size):
    """
    Resize all images in the given directory to the specified output size.

    Parameters:
    - directory: Path to the directory containing the images.
    - output_size: Tuple specifying the desired width and height.
    """
    
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more formats if needed
            filepath = os.path.join(directory, filename)
            
            with Image.open(filepath) as img:
                resized_img = img.resize(output_size, Image.ANTIALIAS)
                resized_img.save(filepath)

            print(f"Resized {filename} to {output_size}")

if __name__ == "__main__":
    directory_path = "labelling/images/city"
    width = 1920
    height = 1080

    resize_images(directory_path, (width, height))
