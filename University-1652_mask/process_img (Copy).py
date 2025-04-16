import os

def get_root_from_image(image_path):
    """
    Extract the root directory from a given image path.
    Assumes the root is the parent of the folder containing the image.
    
    Args:
        image_path (str): Path to an example image file
    
    Returns:
        str: Absolute path to the root directory
    """
    # Convert to absolute path
    abs_image_path = os.path.abspath(image_path)
    
    # Check if the file exists
    if not os.path.exists(abs_image_path):
        raise ValueError(f"Image file '{abs_image_path}' does not exist")
    
    # Get the directory containing the image
    image_dir = os.path.dirname(abs_image_path)
    # Get the parent directory (root)
    root_dir = os.path.dirname(image_dir)
    
    return root_dir

def generate_query_list(root_dir, output_file):
    """
    Generate a text file listing all JPG files under the root directory.
    
    Args:
        root_dir (str): The absolute root directory to scan for images
        output_file (str): The path to the output text file
    """
    if not os.path.exists(root_dir):
        print(f"Error: Root directory '{root_dir}' does not exist")
        return False
    
    try:
        with open(output_file, 'w') as f:
            for dirpath, dirnames, filenames in os.walk(root_dir):
                print(f"Scanning folder: {dirpath}")
                
                jpg_files = [fn for fn in filenames if fn.lower().endswith('.jpg')]
                
                if not jpg_files:
                    print(f"No JPG files found in {dirpath}")
                    continue
                
                for filename in jpg_files:
                    abs_path = os.path.join(dirpath, filename)
                    abs_path = abs_path.replace(os.sep, '/')
                    rel_path = '.' + abs_path[len(os.path.dirname(root_dir)):]
                    f.write(f"{rel_path}\n")
                    print(f"Added: {rel_path}")
        
        print(f"Query list generated successfully: {output_file}")
        return True
    
    except PermissionError:
        print(f"Error: Permission denied when accessing {root_dir} or writing to {output_file}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

def main():
    # Example image path to determine the root
    example_image = "./workshop_gallery_satellite/0b62bsWDQaHOUpSy.jpg"  # Replace with your image path
    output_filename = "query_name.txt"
    
    try:
        # Get the root directory from the example image
        root_directory = get_root_from_image(example_image)
        print(f"Current working directory: {os.getcwd()}")
        print(f"Derived root directory: {root_directory}")
        
        success = generate_query_list(root_directory, output_filename)
        if not success:
            print("Failed to generate query list. Check error messages above.")
        elif os.path.exists(output_filename) and os.path.getsize(output_filename) == 0:
            print("Warning: Generated file is empty. No JPG files found.")
    
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
