from flask import Flask, render_template, request, jsonify
import os
from types import SimpleNamespace
from ocr.inference import TextSystem 
from ocr.inference import process_image
from PIL import Image
from uuid import uuid4

app = Flask(__name__)

def resize_image(input_image_path, output_image_path, base_width=400):
    img = Image.open(input_image_path)
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((base_width, h_size), Image.LANCZOS)
    img.save(output_image_path)

args = SimpleNamespace()

# Setting only the specified attributes as per your request
args.use_mp = False
args.total_process_num = 1

# Assuming you might want to set all other attributes as well, here's how you can do it
args.use_gpu = False
args.use_xpu = False
args.use_npu = False
args.use_mlu = False
args.ir_optim = True
args.use_tensorrt = False
args.min_subgraph_size = 15
args.precision = 'fp32'
args.gpu_mem = 500
args.gpu_id = 0
args.page_num = 0
args.det_algorithm = 'DB'
args.det_model_dir = './Multilingual_PP-OCRv3_det_infer'
args.det_limit_side_len = 960
args.det_limit_type = 'max'
args.det_box_type = 'quad'
args.det_db_thresh = 0.3
args.det_db_box_thresh = 0.6
args.det_db_unclip_ratio = 1.5
args.max_batch_size = 10
args.use_dilation = False
args.det_db_score_mode = 'fast'
args.det_east_score_thresh = 0.8
args.det_east_cover_thresh = 0.1
args.det_east_nms_thresh = 0.2
args.det_sast_score_thresh = 0.5
args.det_sast_nms_thresh = 0.2
args.det_pse_thresh = 0
args.det_pse_box_thresh = 0.85
args.det_pse_min_area = 16
args.det_pse_scale = 1
args.scales = [8, 16, 32]
args.alpha = 1.0
args.beta = 1.0
args.fourier_degree = 5
args.rec_algorithm = 'SVTR_LCNet'
args.rec_model_dir = './content/content/inference_new'
args.rec_image_inverse = True
args.rec_image_shape = '3, 48, 320'
args.rec_batch_num = 6
args.max_text_length = 25
args.rec_char_dict_path = './PaddleOCR/ppocr/utils/dict/latin_dict.txt'
args.use_space_char = True
args.vis_font_path = './doc/fonts/simfang.ttf'
args.drop_score = 0.5
args.e2e_algorithm = 'PGNet'
args.e2e_model_dir = None
args.e2e_limit_side_len = 768
args.e2e_limit_type = 'max'
args.e2e_pgnet_score_thresh = 0.5
args.e2e_char_dict_path = './PaddleOCR/ppocr/utils/ic15_dict.txt'
args.e2e_pgnet_valid_set = 'totaltext'
args.e2e_pgnet_mode = 'fast'
args.use_angle_cls = False
args.cls_model_dir = None
args.cls_image_shape = '3, 48, 192'
args.label_list = ['0', '180']
args.cls_batch_num = 6
args.cls_thresh = 0.9
args.enable_mkldnn = False
args.cpu_threads = 10
args.use_pdserving = False
args.sr_model_dir = None
args.sr_image_shape = '3, 32, 128'
args.sr_batch_num = 1
args.draw_img_save_dir = './inference_results'
args.save_crop_res = False
args.crop_res_save_dir = './output'
args.benchmark = False
args.save_log_path = './log_output/'
args.show_log = True
args.use_onnx = False
args.return_word_box = False
args.process_id = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Check for 'page' in form data
    page = request.form.get('page')
    if page not in ['front', 'back']:
        return jsonify({'error': 'Invalid page parameter. Acceptable values are "front" or "back".'})
    
    # Generate a unique filename using uuid
    # Preserve the file extension
    _, ext = os.path.splitext(file.filename)
    random_filename = str(uuid4()) + ext
    temp_path = os.path.join('uploads', random_filename)
    
    file.save(temp_path)
    
    # Resize the saved image before OCR processing
    if page == 'back':
        resize_image(temp_path, temp_path, 650)
        args.det_db_unclip_ratio = 2.5
        args.use_dilation = True
    else:
        args.det_db_unclip_ratio = 1.5
        args.use_dilation = False
        resize_image(temp_path, temp_path)
    
    text_sys = TextSystem(args)
    # Now process the resized image with OCR
    info_dict = process_image(temp_path, text_sys, args)
    
    # Clean up by removing the temporary file
    os.remove(temp_path)

    # Return the OCR results as JSON
    return jsonify(info_dict)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
