#!/usr/bin/python

import os
import FaceEmotion4Lib.lib_model as mpp
import PIL

class FaceEmotion4Classifier:
    """Class to classify 4 face languages.
    
    The class Emotion4Classifier classify daa in 4 face languages.
    
    Args:
        model_type: Type of model architecture. 
        This can be: 
        'efficientnet_b3', 'inception_resnet_v2', 'inception_v3', 'mobilenet_v3', 'resnet_v2_50', 
        'yolov8n-cls'
        
    Atributos:
        modelo: Model returned by tensorflow.
        model_type: Architecture of model.
        target_size: Input size of model. The input image will be reshape to (target_size,target_size,3).
    """
    def __init__(self,model_type='efficientnet_b3',file_of_weight=''):
        """Inicializer of class FaceEmotion4Classifier.
        
        Args:
            model_type: Type of model architecture. 
            This can be: 
            'efficientnet_b3', 'inception_resnet_v2', 'inception_v3', 'mobilenet_v3', 'resnet_v2_50'
        """
        self.model_type=model_type;

        model_list_hub  = ['efficientnet_b3', 'inception_resnet_v2', 'inception_v3', 'mobilenet_v3', 'resnet_v2_50'];
        
        if len(file_of_weight)>0:
            if   self.model_type in model_list_hub:
                self.model, self.target_size = mpp.create_model(model_type=self.model_type,
                                                                nout=4,
                                                                load_weights=False,
                                                                file_of_weight=file_of_weight);
            
            else:
                print("Don't exist the model:", self.model_type);
                exit();
        
        else:
            if   self.model_type in model_list_hub:
                self.model, self.target_size=mpp.create_model(  model_type=self.model_type,
                                                                nout=4,
                                                                load_weights=True,
                                                                file_of_weight='');
            
            else:
                print("Don't exist the model:", self.model_type);
                exit();
        
    
    def from_img_filepath(self,imgfilepath):
        """Classify a image from the image filepath.
        
        Args:
            imgfilepath: The iamge filepath.
        
        Returns:
            int: The class of image.
        """
        return mpp.evaluate_model_from_file(self.model,imgfilepath, target_size=self.target_size);


    def from_img_pil(self,img_pil):
        """Classify a image from a PIL object.
        
        Args:
            img_pil: The PIL object.
        
        Returns:
            int: The class of image.
        """
        return mpp.evaluate_model_from_pil(self.model,img_pil, target_size=self.target_size);

    def predict_pil(self,img_pil):
        """Classify a image from a PIL object.
        
        Args:
            img_pil: The PIL object.
        
        Returns:
            int: The class of image.
        """
        return mpp.predict_from_pil(self.model,img_pil, target_size=self.target_size);
        
    def target_labels(self):
        """Returns the categories of classifier.
        
        Returns:
            list: The labels of categories resturned by the methods from_img_pil() and from_img_filepath().
        """
        return ['negative','neutro','pain','positive'];


