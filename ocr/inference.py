import os
import re
import sys

# paddle_ocr_dir = '../PaddleOCR'
# sys.path.append(paddle_ocr_dir)

# Get the directory of the current file (__file__ refers to inference.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct an absolute path to the PaddleOCR directory
paddle_ocr_dir = os.path.join(current_dir, '..', 'PaddleOCR')

# Append the PaddleOCR directory to sys.path
sys.path.append(paddle_ocr_dir)

import cv2
import copy
import numpy as np
import json
import time
import logging
from PIL import Image
import json
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image, get_minarea_rect_crop
logger = get_logger()



class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            logger.debug("no valid image provided")
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict
        # else:
        #     logger.debug("dt_boxes num : {}, elapsed : {}".format(
        #         len(dt_boxes), elapse))
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict['cls'] = elapse
            logger.debug("cls num  : {}, elapsed : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        # logger.debug("rec_res num  : {}, elapsed : {}".format(
        #     len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0], rec_result[1]
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

def remove_duplicate_dots(input_string):
    if not isinstance(input_string, (str, bytes)):
        return ""
    return re.sub(r'\.{2,}', '.', input_string)

def extract_info_front(logged_results):
    surname, name, birthdate, expiry_date, card_id, gender, nationality = None, None, None, None, None, None, None
    uppercase_word_pattern = r"^[A-ZŠĐČĆŽ]{3,}$"
    id_pattern = r"^\d{9}$|^[A-Z0-9]{9}$"
    gender_pattern = r"^[A-ZŠĐČĆŽ]{1,2}/[A-ZŠĐČĆŽ]{1,2}$"
    nationality_pattern = r"^[A-Z]{3}$"
    pattern = r"\b\d+(\.\d+)*\b(?![a-zA-Z])"

    # Function to remove matched items by index
    def remove_matched(results, matched_indices):
        return [item for i, item in enumerate(results) if i not in matched_indices]

    # Extracting surname and name
    matched_indices = set()
    for i in range(len(logged_results) - 2):
        text, score = logged_results[i]
        if re.match(uppercase_word_pattern, text) and text != "MNE":
            surname = text
            matched_indices.add(i)

            # Check next item for name match
            next_text, next_score = logged_results[i + 1]
            if re.match(uppercase_word_pattern, next_text) and text != "MNE":
                name = next_text
                matched_indices.add(i + 1)
                break  # Break if the name is immediately after the surname

            # If the immediate next item is not a name, check the item after next
            elif i + 2 < len(logged_results):  # Ensure i + 2 is within bounds
                next_next_text, next_next_score = logged_results[i + 2]
                if re.match(uppercase_word_pattern, next_next_text) and text != "MNE":
                    name = next_next_text
                    matched_indices.add(i + 2)
                    break  # Break if the name is two positions after the surname

    logged_results = remove_matched(logged_results, matched_indices)

    # Match nationality
    matched_indices = set()
    for i, (text, score) in enumerate(logged_results):
        if re.match(nationality_pattern, text):
            nationality = text
            matched_indices.add(i)
            break

    logged_results = remove_matched(logged_results, matched_indices)

    # Match gender
    matched_indices = set()
    for i, (text, score) in enumerate(logged_results):
        if re.match(gender_pattern, text):
            gender = text
            matched_indices.add(i)
            break

    logged_results = remove_matched(logged_results, matched_indices)

    # Match card_id
    matched_indices = set()
    for i, (text, score) in enumerate(logged_results):
        if re.match(id_pattern, text):
            card_id = text
            matched_indices.add(i)
            break

    logged_results = remove_matched(logged_results, matched_indices)

    # Match birthdate
    matched_indices = set()
    for i, (text, score) in enumerate(logged_results):
        if re.match(pattern, text):
            # Prepare the new value to be potentially added to birthdate
            new_birthdate = text if birthdate is None else birthdate + "." + text
            
            # Count the number of digits in the new potential birthdate
            digit_count = len(re.sub(r"[^\d]", "", new_birthdate))
            
            # Count the dots in the new potential birthdate
            dot_count = new_birthdate.count('.')
            
            # Check the position of the last dot
            last_dot_not_at_end = (len(new_birthdate) > new_birthdate.rfind('.') + 1)
            
            if digit_count > 8 or dot_count > 2 or not last_dot_not_at_end:
                # If adding this text makes the digits exceed 8, or there are more than two dots,
                # or the last dot is not at the last place, stop without adding
                break
            else:
                # Update birthdate if it passes the checks
                birthdate = new_birthdate
                matched_indices.add(i)
                
                if digit_count == 8:
                    # If it's exactly 8 digits, we have a complete date, so break
                    break
        else:
            # If a string doesn't match the date pattern and we have already found a part of the birthdate, stop.
            if birthdate is not None:
                break

    logged_results = remove_matched(logged_results, matched_indices)

    # Match expiry_date
    matched_indices = set()
    for i, (text, score) in enumerate(logged_results):
        if re.match(pattern, text):
            # Prepare the new value to be potentially added to birthdate
            new_expirydate = text if expiry_date is None else expiry_date + "." + text
            
            # Count the number of digits in the new potential birthdate
            digit_count = len(re.sub(r"[^\d]", "", new_expirydate))
            
            # Count the dots in the new potential birthdate
            dot_count = new_expirydate.count('.')
            
            # Check the position of the last dot
            last_dot_not_at_end = (len(new_expirydate) > new_expirydate.rfind('.') + 1)
            
            if digit_count > 8 or dot_count > 2 or not last_dot_not_at_end:
                # If adding this text makes the digits exceed 8, or there are more than two dots,
                # or the last dot is not at the last place, stop without adding
                break
            else:
                # Update birthdate if it passes the checks
                expiry_date = new_expirydate
                matched_indices.add(i)
                
                if digit_count == 8:
                    # If it's exactly 8 digits, we have a complete date, so break
                    break
        else:
            # If a string doesn't match the date pattern and we have already found a part of the birthdate, stop.
            if expiry_date is not None:
                break

    logged_results = remove_matched(logged_results, matched_indices)

    info_dict = {
        "surname": surname,
        "name": name,
        "birthdate": birthdate,
        "expiry_date": expiry_date,
        "card_id": card_id,
        "gender": gender,
        "nationality": nationality
    }
    
    return info_dict

def extract_info_back(logged_results):
    jmbg, expiry_date, city, line1, line2, line3 = None, None, None, None, None, None
    jmbg_pattern = re.compile(r'\b\d{13}\b')
    pattern = r"\b\d+(\.\d+)*\b(?![a-zA-Z])"

    # Function to remove matched items by index
    def remove_matched(results, matched_indices):
        return [item for i, item in enumerate(results) if i not in matched_indices]

    matched_indices = set()

    # Iterate over logged_results to find indices of elements with lowercase letters
    for i, (text, _) in enumerate(logged_results):
        if any(char.islower() for char in text):
            matched_indices.add(i)
    
    # Use remove_matched to filter out these elements
    logged_results = remove_matched(logged_results, matched_indices)

    matched_indices = set()
    # Search for the JMBG in the filtered logged_results
    for i, (text, _) in enumerate(logged_results):
        if jmbg_pattern.match(text):
            jmbg = text  # Assign the matched JMBG
            matched_indices.add(i)  # Add index of JMBG to the set of indices to remove
            break  # Assuming only one JMBG per logged_results, break after finding

    # Use remove_matched to filter out matched elements, including the JMBG
    logged_results = remove_matched(logged_results, matched_indices)

    # Match expiry_date
    matched_indices = set()
    for i, (text, score) in enumerate(logged_results):
        if re.match(pattern, text):
            # Prepare the new value to be potentially added to birthdate
            new_expirydate = text if expiry_date is None else expiry_date + "." + text
            
            # Count the number of digits in the new potential birthdate
            digit_count = len(re.sub(r"[^\d]", "", new_expirydate))
            
            # Count the dots in the new potential birthdate
            dot_count = new_expirydate.count('.')
            
            # Check the position of the last dot
            last_dot_not_at_end = (len(new_expirydate) > new_expirydate.rfind('.') + 1)
            
            if digit_count > 8 or dot_count > 2 or not last_dot_not_at_end:
                # If adding this text makes the digits exceed 8, or there are more than two dots,
                # or the last dot is not at the last place, stop without adding
                break
            else:
                # Update birthdate if it passes the checks
                expiry_date = new_expirydate
                matched_indices.add(i)
                
                if digit_count == 8:
                    # If it's exactly 8 digits, we have a complete date, so break
                    break
        else:
            # If a string doesn't match the date pattern and we have already found a part of the birthdate, stop.
            if expiry_date is not None:
                break

    logged_results = remove_matched(logged_results, matched_indices)

    matched_indices = set()
    # Second pass: Match and remove the city element
    for i, (text, _) in enumerate(logged_results):
        if text.startswith("PJ"):
            city = "PJ " + text[2:]  # Format the city element
            matched_indices.add(i)  # Add index of city to the set of indices to remove
            break  # Assuming only one city element is present

    logged_results = remove_matched(logged_results, matched_indices)

    # Extract last three elements for line1, line2, line3 if they exist
    if len(logged_results) >= 3:
        line1, line2, line3 = logged_results[-3][0], logged_results[-2][0], logged_results[-1][0]
    elif len(logged_results) == 2:
        line1, line2 = logged_results[-2][0], logged_results[-1][0]
    elif len(logged_results) == 1:
        line1 = logged_results[-1][0]

    def sanitize_line(line):
        if line is None:
            return ''
        # Replace all 'O' with '0'
        return line.replace('O', '0')


    def post_process_line1(line, jmbg):
        # Sanitize line1 to replace 'O' with '0'
        sanitized_line = sanitize_line(line)
        jmbg = jmbg or ''
        jmbg_index = sanitized_line.find(jmbg)
        if jmbg_index != -1:
            # JMBG found; replace everything after JMBG with '<'
            sanitized_line = sanitized_line[:jmbg_index + len(jmbg)] + ('<' * (30 - (jmbg_index + len(jmbg))))
        else:
            sanitized_line = sanitized_line[:30]
        
        # Ensure the line is exactly 30 characters
        return sanitized_line.ljust(30, '<')[:30]

    def post_process_line2_and_line3(line, is_line2=False):
        # Sanitize line for 'O' to '0' replacement if line2
        sanitized_line = sanitize_line(line) if is_line2 else line

        # Find the index of the last uppercase letter
        last_upper_index = next((i for i, char in reversed(list(enumerate(sanitized_line))) if char.isupper()), None)
        if last_upper_index is not None:
            if is_line2:
                # Ensure proper length and format for line2
                sanitized_line = sanitized_line[:last_upper_index + 1] + ('<' * (28 - last_upper_index)) + sanitized_line[-1]
            else:
                # Process line3 similarly, but without the digit constraint
                sanitized_line = sanitized_line[:last_upper_index + 1] + ('<' * (29 - last_upper_index))
        
        # Ensure the final character for line2 is a digit; replace if not
        if is_line2 and not sanitized_line[-1].isdigit():
            sanitized_line = sanitized_line[:-1] + '0'

        # Ensure the line is exactly 30 characters
        return sanitized_line.ljust(30, '<')[:30]

    line1 = post_process_line1(line1, jmbg)
    line2 = post_process_line2_and_line3(line2, is_line2=True)
    line3 = post_process_line2_and_line3(line3)

    info_dict = {
        "jmbg": jmbg,
        "expiry_date": expiry_date,
        "city": city,
        "line1": line1,
        "line2": line2,
        "line3": line3
    }
    
    return info_dict

def process_image(image_path, text_sys, args):
    font_path = args.vis_font_path
    drop_score = args.drop_score

    start_time = time.time()

    img, flag_gif, flag_pdf = check_and_read(image_path)
    if not flag_gif and not flag_pdf:
        img = cv2.imread(image_path)
    if img is None:
        logger.debug(f"error in loading image: {image_path}")
        return None
    
    dt_boxes, rec_res, time_dict = text_sys(img)
    elapsed_time = time.time() - start_time
    logger.info(f"Processing time for {image_path}: {elapsed_time:.3f}s")
    print(rec_res)
    
    if (args.det_db_unclip_ratio == 2.5):
        extracted_info = extract_info_back(rec_res)
    else:
        extracted_info = extract_info_front(rec_res)

    return extracted_info

