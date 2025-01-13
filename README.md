# cnn_face_emotion
cnn_face_emotion


# Using library
Since the code uses an old version of keras, it needs to be placed at the beginning of the main.py code.

    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'

    import FaceEmotion4Lib.Classifier as fec
    from PIL import Image

    cls=fec.FaceEmotion4Classifier(model_type='efficientnet_b3');

    img_pil = Image.new('RGB', (400,300), 'white');

    res=cls.from_img_pil(img_pil);

    print(res);

# Installation summary - Dataset BER2024

    git clone https://github.com/trucomanx/cnn_face_emotion
    gdown 10PZUfBSJt3FXcNaA8UfvP6hGC46E0NoR
    unzip models.zip -d cnn_face_emotion/library/FaceEmotion4Lib/models
    cd cnn_face_emotion/library
    python3 setup.py sdist
    pip3 install dist/FaceEmotion4Lib-*.tar.gz

# Installation summary - Dataset FULL2024

    git clone https://github.com/trucomanx/cnn_face_emotion
    gdown 18ZTsD3FF0_1H3goacGPZwOgcLOKXhw0b
    unzip models_face_full.zip -d cnn_face_emotion/library/FaceEmotion4Lib/models
    cd cnn_face_emotion/library
    python3 setup.py sdist
    pip3 install dist/FaceEmotion4Lib-*.tar.gz
    

