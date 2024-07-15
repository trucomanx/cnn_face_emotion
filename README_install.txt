




# Packaging


Download the source code

    
    git clone https://github.com/trucomanx/cnn_face_emotion


Download the models.
    
    gdown 10PZUfBSJt3FXcNaA8UfvP6hGC46E0NoR
    unzip models.zip -d cnn_face_emotion/library/FaceEmotion4Lib/models
    

The next command generates the `dist/FaceEmotion4Lib-VERSION.tar.gz` file.

    cd cnn_face_emotion/library
    python3 setup.py sdist

For more informations use `python setup.py --help-commands`

# Install 

Install the packaged library

    pip3 install dist/FaceEmotion4Lib-*.tar.gz

# Uninstall

    pip3 uninstall FaceEmotion4Lib
