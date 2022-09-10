import zlib
import zipfile

def compress(file_names):
    print("File Paths:")
    print(file_names)

    path = "../ImageClassifier/"

    # Select the compression mode ZIP_DEFLATED for compression
    # or zipfile.ZIP_STORED to just store the file
    compression = zipfile.ZIP_DEFLATED

    # create the zip file first parameter path/name, second mode
    zf = zipfile.ZipFile("final_project.zip", mode="w")
    try:
        for file_name in file_names:
            # Add file to the zip file
            # first parameter file to zip, second filename in zip
            zf.write(path + file_name, file_name, compress_type=compression)

    except FileNotFoundError:
        print("An error occurred")
    finally:
        # Don't forget to close the file!
        zf.close()


file_names= ["Image Classifier Project.html", "utility_functions.py", "train.py", "predict.py", "model_functions.py"]
compress(file_names)
