# import os
# import glob
# import time
# import tensorflow as tf
# import sys
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import warnings
# import cv2
# import itertools 
# from object_detection.utils import label_map_util
# from object_detection.utils import config_util
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.builders import model_builder
# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'
# import cv2
# import numpy as np
# from PIL import Image
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# import pytesseract
# # import argparse
# import cv2
# import os
# from spellchecker import SpellChecker



# # path = './workspace/training_demo/images/test/lemos'



# def load_model():
#     PATH_TO_LABELS = 'C:/Users/Divya/OneDrive/Desktop/teddy/models/label_map.pbtxt'
#     LABEL_FILENAME = 'label_map.pbtxt'

#     PATH_TO_CFG = 'C:/Users/Divya/OneDrive/Desktop/teddy/models/pipeline.config'
#     PATH_TO_CKPT = 'C:/Users/Divya/OneDrive/Desktop/teddy/models/checkpoint'

#     print('Loading model... ', end='')
#     start_time = time.time()

#     # Load pipeline config and build a detection model
#     configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
#     model_config = configs['model']
#     detection_model = model_builder.build(model_config=model_config, is_training=False)

#     # Restore checkpoint
#     ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
#     ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

# @tf.function
# def detect_fn(image):
#     """Detect objects in image."""

#     image, shapes = detection_model.preprocess(image)
#     prediction_dict = detection_model.predict(image, shapes)
#     detections = detection_model.postprocess(prediction_dict, shapes)

#     return detections

# def load_image_into_numpy_array(path):
#     return np.array(Image.open(path))

# def read_class_names(class_file_name):
#     # loads class name from a file
#     names = {}
#     with open(class_file_name, 'r') as data:
#         for ID, name in enumerate(data):
#             names[ID] = name.strip('\n')
#     return names

# def convert_image_dtype(img, target_type_min, target_type_max, target_type):
#     imin = img.min()
#     imax = img.max()

#     a = (target_type_max - target_type_min) / (imax - imin)
#     b = target_type_max - a * imax
#     new_img = (a * img + b).astype(target_type)
#     return new_img

# #pre processing functions
# # get grayscale image
# def get_grayscale(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # noise removal
# def remove_noise(image):
#     return cv2.medianBlur(image,5)
 
# #thresholding
# def thresholding(image):
#     return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# #dilation
# def dilate(image):
#     kernel = np.ones((5,5),np.uint8)
#     return cv2.dilate(image, kernel, iterations = 1)
    
# #erosion
# def erode(image):
#     kernel = np.ones((3,3),np.uint8)
#     return cv2.erode(image, kernel, iterations = 1)

# #opening - erosion followed by dilation
# def opening(image):
#     kernel = np.ones((3,3),np.uint8)
#     return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# #canny edge detection
# def canny(image):
#     return cv2.Canny(image, 100, 200)

# #skew correction
# def deskew(image):
#     coords = np.column_stack(np.where(image > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return rotated

# #template matching
# def match_template(image, template):
#     return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# #Split string into characters 
# def split(word): 
#     return [char for char in word]  


# def is_valid_character(ch,object_type):
#     if(object_type==1):
#         if ch in CHARACTERLIST_FOR_ALPHA_NUMERIC:
#             return 1
#         else:
#             return 0
#     elif(object_type==2):
#         if ch in CHARACTERLIST_FOR_AMOUNT_NUMBER:
#             return 1
#         else:
#             return 0
#     elif(object_type==3):
#         if ch in CHARACTERLIST_FOR_DATE:
#             return 1
#         else:
#             return 0    
#     elif(object_type in (4,5,6)):
#         if ch in CHARACTERLIST_FOR_NUMERIC:
#             return 1
#         else:
#             return 0     
        

# def extract_only_valid_characters(text,object_class):
#     #filter non-alphanumeric chracters
#     print('\n\nfiltering test')
#     words = text.split()
#     print(words)
#     print('filtering test\n\n')
    
#     final_text = ""
#     for word in words:
#         word_1 = split(word)
#         print('word_1')
#         print(word_1)
#         new_word = ''
#         for c in word_1:
#             if(is_valid_character(c,object_class) == 1): # checking if indiviual chr is alphanumeric or not
#                 new_word = new_word + c    
#         final_text = final_text+ " " +new_word
        
#     #return read_text
#     print('\n\nfinal_text test')
#     print(final_text)
#     print('final_text test\n\n')
        
#     return final_text

# def word_filtering_amount_word(text,object_class):
#     text = text.replace("-", " ")
#     words = text.split()
#     full_text = ""
#     misspelled = spell.unknown(words)
#     print(misspelled)
    
#     for word in misspelled:
#         print(spell.correction(word))
#         word1 = spell.correction(word)
#         if word1 is not None:
#           text.replace(word, word1)
        
#     words = text.split()    
#     for word in words:
#         print(spell.correction(word))
#         word = spell.correction(word)
#         if word is not None:
#           if word.lower() in WORD_LIST_FOR_AMOUNT_WORD:
#             full_text = full_text + word+" "
#     print(full_text)
    
#     return full_text

# def ocr_further_processing(img,roi_boundary,base_image_width,object_class): #image and a tuple

#     (_, thresh) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     unsharp_image = thresh

#     plt.figure()
#     # plt.imshow(cv2.cvtColor(unsharp_image, cv2.COLOR_GRAY2RGB))
#     plt.imshow(unsharp_image)
#     plt.show()
        
#     # kernel = np.ones((3,3),np.uint8)
#     # eroded = cv2.erode(unsharp_image,kernel,iterations = 1) #white pixels gets eroded
    
#     plt.figure()
#     plt.imshow(thresh)
#     plt.show()
#     return thresh

# def read_text_using_ocr(image,ROI,object_class = 3):  #{1:'amountWord'}, 2:'amountNumber'}, 3: 'date'}, 4:'validPeriod'}, 5: 'ABArouting'}, 6: 'signature'
    
#     gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     # (thresh, gray) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     # (_, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     (y1,x1,y2,x2) = ROI
#     bound = [y1,x1,y2,x2]
    
#     # ROI_12 = thresh[y1:y2,x1:x2] #important
#     ROI_12 = gray[y1:y2,x1:x2] #important
    
#     # removing grid like distortions
#     ROI_12 = ocr_further_processing(ROI_12,bound,image.shape[1],object_class)
	
#     #custom_config = r'--oem 3 --psm 6'
#     custom_config = r'--oem 3 --psm 12'
  
#     if object_class == 1:
#         # custom_config = r'--oem 3 --psm 12 -l eng'
#         custom_config = r'--oem 3 --psm 12'
#         read_text = pytesseract.image_to_string(ROI_12, config=custom_config)
#     elif object_class ==2:
#         # custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789,.$'
#         custom_config = r'--oem 3  --psm 12'
#         read_text = pytesseract.image_to_string(ROI_12, config=custom_config)
#     elif object_class ==3:
#         # custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789,.$'
#         custom_config = r'--oem 3  --psm 12'
#         read_text = pytesseract.image_to_string(ROI_12, config=custom_config)
#     else:
#         read_text = pytesseract.image_to_string(ROI_12, config=custom_config)
#     # read_text = pytesseract.image_to_string(ROI_12, config=custom_config)
    
#     final_text = extract_only_valid_characters(read_text,object_class)
    
#     if object_class == 1:
#         final_text = word_filtering_amount_word(final_text,1)
    
#     pltImage = np.copy(ROI_12) 
#     pltImage = cv2.cvtColor(pltImage, cv2.COLOR_BGR2RGB) 
#     plt.figure()
   
#     plt.imshow(pltImage)
#     plt.text(0.1, 0.9,final_text, size=15, color='red')
#     plt.show()
    
#     imFileName = "test_{}.jpg".format(object_class)
    
#     # cv2.imwrite(imFileName,pltImage)
    
#     return final_text
    
# def printed_cheque_data_extraction(input_image):
#     load_model()
#     im = Image.open(input_image)
#     resized_img = im.resize((2365, 1100))
#     resized_img.save('C:/Users/Divya/OneDrive/Desktop/teddy/static/check_images/resized_input.jpg')
#     spell = SpellChecker(distance=1)  # set at initialization
#     FILTERING_THRESH = 0.3

#     CLASSES = {1: {'id': 1, 'name': 'amountWord'}, 2: {'id': 2, 'name': 'amountNumber'}, 3: {'id': 3, 'name': 'date'}, 4: {'id': 4, 'name': 'validPeriod'}, 5: {'id': 5, 'name': 'ABArouting'}, 6: {'id': 6, 'name': 'signature'}}

#     path = 'C:/Users/Divya/OneDrive/Desktop/teddy/static/test'
#     out_path = 'C:/Users/Divya/OneDrive/Desktop/teddy/static/printed_cheque_results'
#     attempt = '_i'
#     files = os.listdir(path)

#     image_paths = []

#     for f in glob.glob(path+'/*.jpg'):
#         out_filename = out_path+f[f.find('test/')+4:f.find('.jpg')]+attempt+'.jpg'
#         print("image path : "+f)
#         image_paths.append(str(f))
#         print("output path : "+out_filename)
#         print(image_paths)

#     IMAGE_PATHS = image_paths

#     print(IMAGE_PATHS)
#     #colors
#     AMOUNTWORD_C = (24,24,169)
#     AMOUNTNUMBER_C = (233,82,82)
#     DATE_C = (112,8,8)
#     VALIDPERIOD_C = (193,24,193)
#     ABAROUTING_C = (63,169,9)
#     SIGNATURE_C = (21,218,249)

#     #detection filtering threshold values
#     AMOUNT_WORD_THRESH = 0.25
#     AMOUNT_NUMMBER_THRESH = 0.5
#     DATE_THRESH = 0.5
#     VALID_PERIOD_THRESH = 0.5
#     ABA_THRESH = 0.5
#     SIGNATURE_THRESH = 0.5


#     CHARACTERLIST_FOR_ALPHA_NUMERIC = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","Y","X","Y","Z",\
#                     "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","y","x","y","z",\
#                     "1","2","3","4","5","6","7","8","9","0"]

#     CHARACTERLIST_FOR_AMOUNT_WORD = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","Y","W","X","Y","Z",\
#                     "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","y","w","x","y","z",\
#                     "1","2","3","4","5","6","7","8","9","0","-","—","_"]

#     CHARACTERLIST_FOR_DATE = ["1","2","3","4","5","6","7","8","9","0","-","/","l"]

#     # CHARACTERLIST_FOR_DATE = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","Y","X","Y","Z",\
#     #                 "1","2","3","4","5","6","7","8","9","0","-","/","l"]

#     CHARACTERLIST_FOR_NUMERIC = ["1","2","3","4","5","6","7","8","9","0"]

#     CHARACTERLIST_FOR_AMOUNT_NUMBER = ["1","2","3","4","5","6","7","8","9","0",".",",","-"]

#     CHARACTERLIST_FOR_ALPHA_ONLY = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","Y","X","Y","Z",\
#                     "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","y","x","y","z"]

#     CHARACTERLIST = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","Y","X","Y","Z",\
#                     "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","y","x","y","z",\
#                     "1","2","3","4","5","6","7","8","9","0","-",".",",","/","_","="]
        
#     WORD_LIST_FOR_AMOUNT_WORD = ["and","eight","eighteen","eighty","eleven","fifteen","fifty","five","forty",\
#                                 "four","fourteen","hundred","hundredth","million","nine","nineteen","ninety",\
#                                     "one","seven","seventeen","seventy","six","sixteen","sixty","ten","thirteen",\
#                                         "thirty","thousand","three","trillion","twelve","twenty","two","dollars","cents"]



#     spell.word_frequency.load_words(WORD_LIST_FOR_AMOUNT_WORD)
#     all_results = {}  
#     all_components = {}
#     all_read_texts = {}
#     lemos_index=0 #number of detected components
    
#     amountWords = {}
#     amountNumbers = {}
#     dates = {}
#     validPeriods = {}
#     ABAroutings = {}
#     signatures = {}
#     category_index = label_map_util.create_category_index_from_labelmap("/content/bankchecks/workspace/training_demo/annotations/label_map.pbtxt",
#                                                                     use_display_name=True)
#     print(category_index)

#     i = 0
#     all_image_inferences = {}
#     for image_path in IMAGE_PATHS:
#         #here image_name is the ma,e of the image without file extension (file type) 
#         image_name = image_path[image_path.rfind('/')+1:image_path.rfind('.')]

#         print(image_name)

#         print('Running inference for {}... '.format(image_path), end='')

#         image_np = load_image_into_numpy_array(image_path)

#         image_height = image_np.shape[0] #multiply with xmin and xmax
#         image_width = image_np.shape[1] #multiply with ymin and ymax

#         input_tensor = tf.convert_to_tensor(value=np.expand_dims(image_np, 0), dtype=tf.float32)

#         detections = detect_fn(input_tensor)

#         # All outputs are batches tensors.
#         # Convert to numpy arrays, and take index [0] to remove the batch dimension.
#         # We're only interested in the first num_detections.
#         num_detections = int(detections.pop('num_detections'))
#         detections = {key: value[0, :num_detections].numpy()
#                     for key, value in detections.items()}
#         detections['num_detections'] = num_detections

#         # detection_classes should be ints.
#         detections['detection_classes'] = detections['detection_classes'].astype(np.uint8)


#         print(len(detections['detection_boxes']))
#         # since classes detected based on 0 base have to add the label_id_offset to the detected number
#         label_id_offset = 1
#         image_np_with_detections = image_np.copy()

#         viz_utils.visualize_boxes_and_labels_on_image_array(
#                 image_np_with_detections,
#                 detections['detection_boxes'],
#                 detections['detection_classes']+label_id_offset,
#                 detections['detection_scores'],
#                 category_index,
#                 use_normalized_coordinates=True,
#                 max_boxes_to_draw=200,
#                 min_score_thresh=.25,
#                 agnostic_mode=False)

        
#         # textFileName3 = output_path[:output_path.find('.jpg')]+"_3.txt"
#         output_path_text_file = out_path+'/'+image_name+'_att_1.txt'
        
#         bboxes_list = []

#         with open(output_path_text_file, 'w') as filehandle:
#             for (detec,cls,scr) in zip(detections['detection_boxes'],detections['detection_classes'],detections['detection_scores']):
#                 # [xmin, ymin, xmax, ymax, class, prob.score]
            
#                 # list_temp = [int(detec[0]*image_width),int(detec[1]*image_height),int(detec[2]*image_width),int(detec[3]*image_height),cls,scr]
#                 # list_temp = [np.int64(detec[0]*image_width),np.int64(detec[1]*image_height),np.int64(detec[2]*image_width),np.int64(detec[3]*image_height),cls,scr]
#                 # [ymin, xmin, ymax, xmax, class, prob.score]
#                 list_temp = [np.int64(detec[0]*image_height),np.int64(detec[1]*image_width),np.int64(detec[2]*image_height),np.int64(detec[3]*image_width),cls,scr]
                
#                 bboxes_list.append(list_temp)
#                 # print(np.int64(detec[0]*image_width))

#                 textLine = str(list_temp[0])+" " +str(list_temp[1])+" " +str(list_temp[2])+" " +str(list_temp[3])+" " +str(scr)+" " +str(cls)
#                 filehandle.write('%s\n' % textLine)

#         all_image_inferences.update({image_name:bboxes_list})

#         # print(image_np_with_detections)
#         plt.figure()
#         plt.imshow(image_np_with_detections)
#         # plt.savefig('test1234{}_.png'.format(i),dpi=300)
#         # output_path_ = out_path+'/'+image_name+'att{}_.png'.format(i)
#         output_path_ = out_path+'/'+image_name+'att_1.png'
#         print(output_path_)
#         viz_utils.save_image_array_as_png(image_np_with_detections,output_path_)
#         # cv2.imwrite('test1234{}.jpg'.format[i],image_np_with_detections)
#         print('Done')
#         i +=1
#         plt.show()

#     print(all_image_inferences.keys())

#     print(all_image_inferences)

#     print(CLASSES)
#     filename = 'C:/Users/Divya/OneDrive/Desktop/teddy/static/check_images/resized_input.jpg'
#     test1 = cv2.imread(filename,cv2.IMREAD_COLOR)

#     #for the ease of ocr process make a permannet copy of the image
#     test1copy = np.copy(test1)
    
#     #extracting bboxes details into a list of lists
#     bboxes_list = all_image_inferences[input_image]
    
#     print(bboxes_list)
    
#     #producing dictionries for seperate classes
#     for item in bboxes_list:
#         category_index = int(item[4]+1)
#         #CLASSES = {1: {'id': 1, 'name': 'amountWord'}, 2: {'id': 2, 'name': 'amountNumber'}, 3: {'id': 3, 'name': 'date'}, 4: {'id': 4, 'name': 'validPeriod'}, 5: {'id': 5, 'name': 'ABArouting'}, 6: {'id': 6, 'name': 'signature'}}
#         # category_index = item[4]+1   # 0 based nisa
#         #ymin,xmin,ymax,xmax,probabality_score,category_index,category
#         coord_and_class = [item[0],item[1],item[2],item[3],item[5],category_index,CLASSES[category_index]['name']]
                
#         #--- amountWord --- #
#         if category_index == 1: #amountWord
#             amountWord_id = 'aW_1'
#             #filtering the detected bounding boxes using a pre deifened threshold value
#             if float(item[5])>AMOUNT_WORD_THRESH:
#                 if amountWords=={}:
#                     amountWords.update({amountWord_id:coord_and_class})
#                     all_components.update({'amountWord':coord_and_class})
#                     lemos_index+=1
#                     # print(amountWords)
#                 else:
#                     if float(item[5])>float(amountWords[amountWord_id][4]):
#                         # print("new amountWord")
#                         amountWords.update({amountWord_id:coord_and_class})
#                         all_components.update({'amountWord':coord_and_class})
#                         lemos_index+=1
                        
        
#         #--- amountNumber --- #
#         if category_index == 2: #amountNumber
#             amountNumber_id = 'aN_1'
#             #filtering the detected bounding boxes using a pre deifened threshold value
#             if float(item[5])>AMOUNT_NUMMBER_THRESH:
#                 if amountNumbers=={}:
#                     amountNumbers.update({amountNumber_id:coord_and_class})
#                     all_components.update({'amountNumber':coord_and_class})
#                     lemos_index+=1
#                     # print(amountNumbers)
#                 else:
#                     if float(item[5])>float(amountNumbers[amountNumber_id][4]):
#                         # print("new amountWord")
#                         amountNumbers.update({amountNumber_id:coord_and_class})
#                         all_components.update({'amountNumber':coord_and_class})
#                         lemos_index+=1
                        
#         #--- date --- #
#         if category_index == 3: #date
#             date_id = 'dt_1'
#             #filtering the detected bounding boxes using a pre deifened threshold value
#             if float(item[5])>DATE_THRESH:
#                 if dates=={}:
#                     dates.update({date_id:coord_and_class})
#                     all_components.update({'date':coord_and_class})
#                     lemos_index+=1
#                     # print(dates)
#                 else:
#                     if float(item[5])>float(dates[date_id][4]):
#                         # print("new amountWord")
#                         dates.update({date_id:coord_and_class})
#                         all_components.update({'date':coord_and_class})
#                         lemos_index+=1
        
#         #--- validPeriod --- #
#         if category_index == 4: #validPeriod
#             validPeriod_id = 'vp_1'
#             #filtering the detected bounding boxes using a pre deifened threshold value
#             if float(item[5])>VALID_PERIOD_THRESH:
#                 if validPeriods=={}:
#                     validPeriods.update({validPeriod_id:coord_and_class})
#                     all_components.update({'validPeriod':coord_and_class})
#                     lemos_index+=1
#                     # print(validPeriods)
#                 else:
#                     if float(item[5])>float(validPeriods[validPeriod_id][4]):
#                         # print("new amountWord")
#                         validPeriods.update({validPeriod_id:coord_and_class})
#                         all_components.update({'validPeriod':coord_and_class})
#                         lemos_index+=1
                        
#         #--- ABArouting --- #
#         if category_index == 5: #ABArouting
#             ABArouting_id = 'aba_1'
#             #filtering the detected bounding boxes using a pre deifened threshold value
#             if float(item[5])>ABA_THRESH:
#                 if ABAroutings=={}:
#                     ABAroutings.update({ABArouting_id:coord_and_class})
#                     all_components.update({'ABArouting':coord_and_class})
#                     lemos_index+=1
#                     # print(ABAroutings)
#                 else:
#                     if float(item[5])>float(ABAroutings[ABArouting_id][4]):
#                         # print("new ABArouting")
#                         ABAroutings.update({ABArouting_id:coord_and_class})
#                         all_components.update({'ABArouting':coord_and_class})
#                         lemos_index+=1
                        
#         #--- signature --- #
#         if category_index == 6: #signature
#             signature_id = 'sg_1'
#             #filtering the detected bounding boxes using a pre deifened threshold value
#             if float(item[5])>SIGNATURE_THRESH:
#                 if signatures=={}:
#                     signatures.update({signature_id:coord_and_class})
#                     all_components.update({'signature':coord_and_class})
#                     lemos_index+=1
#                     # print(signatures)
#                 else:
#                     if float(item[5])>float(signatures[signature_id][4]):
#                         # print("new amountWord")
#                         signatures.update({signature_id:coord_and_class})
#                         all_components.update({'signature':coord_and_class})
#                         lemos_index+=1
        
#     #print(mileposts.get('mp_1')[3])
#     #print(mileposts['mp_1'][3])
    
#     print("\n\nprinting classes seperately\n")
                
#     print(amountWords)
#     print(amountNumbers)
#     print(dates)
#     print(validPeriods)
#     print(ABAroutings)
#     print(signatures)
    
#     print("\n\nAll Components\n")
#     print(all_components) 
    
#     print("\n\nNumber of items detected is --- {}\n".format(lemos_index))     
    
    
#     for component in all_components.values():
#         ##tensorflow provides cordinates of the format ymin=bbox[0],xmin,ymax,xmax
#         (y1, x1), (y2, x2) = (component[0], component[1]), (component[2], component[3])
#         ##cv2.rectangle(test1, (x1, y1), (x2, y2), (0,100,200), -1)
        
#         print(component)
               
#         if(component[5]==1):
#             cv2.rectangle(test1, (x1, y1), (x2, y2), AMOUNTWORD_C, 4)#b,g,r
#             cv2.putText(test1,'{} : {}%'.format(component[6],round(component[4]*100,2)),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,AMOUNTWORD_C,2)
#             read_text_2 = read_text_using_ocr(test1copy,[y1, x1, y2, x2],1)
#             read_text_2 = read_text_2.strip()
#             all_read_texts.update({'amountWord':read_text_2})
#             print("Read text amountWord: --- {}\n".format(read_text_2))
                        
#         elif(component[5]==2):
#             cv2.rectangle(test1, (x1, y1), (x2, y2), AMOUNTNUMBER_C, 4)
#             cv2.putText(test1,'{} : {}%'.format(component[6],round(component[4]*100,2)),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,AMOUNTNUMBER_C,2)
#             read_text_2 = read_text_using_ocr(test1copy,[y1, x1, y2, x2],2)
#             read_text_2 = read_text_2.strip().replace(" ", "")
#             all_read_texts.update({'amountNumber':read_text_2})
#             print("Read text amountNumber: --- {}\n".format(read_text_2))
            
#         elif(component[5]==3):
#             cv2.rectangle(test1, (x1, y1), (x2, y2), DATE_C, 4)
#             cv2.putText(test1,'{} : {}%'.format(component[6],round(component[4]*100,2)),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,DATE_C,2)
#             read_text_2 = read_text_using_ocr(test1copy,[y1, x1, y2, x2],3)
#             read_text_2 = read_text_2.strip().replace(" ", "").replace("l", "1")
#             all_read_texts.update({'date':read_text_2})
#             print("Read text date: --- {}\n".format(read_text_2))
            
#         elif(component[5]==4):
#             cv2.rectangle(test1, (x1, y1), (x2, y2), VALIDPERIOD_C, 4)
#             cv2.putText(test1,'{} : {}%'.format(component[6],round(component[4]*100,2)),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,VALIDPERIOD_C,2)
#             read_text_2 = read_text_using_ocr(test1copy,[y1, x1, y2, x2],4)
#             read_text_2 = read_text_2.strip()
#             all_read_texts.update({'validPeriod':read_text_2})
#             print("Read text validPeriod: --- {}\n".format(read_text_2))
            
#         elif(component[5]==5):
#             cv2.rectangle(test1, (x1, y1), (x2, y2), ABAROUTING_C, 4)
#             cv2.putText(test1,'{} : {}%'.format(component[6],round(component[4]*100,2)),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,ABAROUTING_C,2)
#             read_text_2 = read_text_using_ocr(test1copy,[y1, x1, y2, x2],5)
#             read_text_2 = read_text_2.strip().replace(" ", "")
#             all_read_texts.update({'ABArouting':read_text_2})
#             print("Read text ABArouting: --- {}\n".format(read_text_2))
            
#         elif(component[5]==6):
#             cv2.rectangle(test1, (x1, y1), (x2, y2), SIGNATURE_C, 4)
#             cv2.putText(test1,'{} : {}%'.format(component[6],round(component[4]*100,2)),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,SIGNATURE_C,2)
#             read_text_2 = read_text_using_ocr(test1copy,[y1, x1, y2, x2],6)
#             read_text_2 = read_text_2.strip().replace(" ", "")
#             all_read_texts.update({'signature':read_text_2})
#             print("Read text signature: --- {}\n".format(read_text_2))            

#     #result_final = {}


#     ##new file name for the output image
#     filename1 = filename[:filename.find('.')]+'_Lemos_1'+ filename[filename.find('.'):] 
    
#     ##csv file name
#     #filename1_csv = filename[:filename.find('.')]+'_LemosTest_3_further_ii'+'.csv'
    
#     print(filename1)
#     cv2.imwrite(filename1,test1)
    
    
#     print("\nresult_final\n")
#     print(all_read_texts)
#     print("\nresult_final\n")
#     all_results.update({input_image_:all_read_texts}) 
#     print(IMAGE_PATHS)
#     for filename_ in IMAGE_PATHS:
#         input_image = filename_[filename_.rfind('/')+1:filename_.rfind('.')]
#         print(input_image)    
#         main_(filename_,input_image)

#     print(all_results.keys())

#     for result in all_results.keys():
#         print(all_results[result])
#     print(all_results)


# def pdf_to_image_for_printed_cheques(file):
#     pdf_file = fitz.open(file)
#     print(file)
#     for page_index in range(len(pdf_file)):
#         # get the page itself
#         page = pdf_file[page_index]
#         # get image list
#         image_list = page.get_images()
#         # printing number of images found in this page
#         if image_list:
#             print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
#         else:
#             print("[!] No images found on page", page_index)
#         for image_index, img in enumerate(image_list, start=1):
#             # get the XREF of the image
#             xref = img[0]
#             # extract the image bytes
#             base_image = pdf_file.extract_image(xref)
#             image_bytes = base_image["image"]
#             # get the image extension
#             image_ext = base_image["ext"]
#             # load it to PIL
#             image = Image.open(io.BytesIO(image_bytes))
#             # save it to local disk
#             image.save(open(f"C:/Users/Divya/OneDrive/Desktop/teddy/static/printed_check_images/input_handwritten_cheques.{image_ext}", "wb"))
#             printed_cheque_data_extraction("C:/Users/Divya/OneDrive/Desktop/teddy/static/printed_check_images/input_handwritten_cheques.jpeg")
