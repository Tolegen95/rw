from flask import Flask, request, render_template, send_file, jsonify, session
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
from pathlib import Path
import glob
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Для работы с сессиями

# Конфигурация
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results_web'

# Словари с переводами
TRANSLATIONS = {
    'ru': {
        'title': 'Распознавание объектов на ж/д',
        'model': 'Лучшая обученная модель',
        'conf_threshold': 'Порог уверенности (0-1):',
        'iou_threshold': 'Порог IoU (0-1):',
        'drop_files': 'Перетащите файлы сюда или',
        'select_files': 'Выберите файлы',
        'start_detection': 'Начать распознавание',
        'clear_all': 'Очистить все',
        'processing': 'Обработка изображений...',
        'results': 'Результаты распознавания',
        'original': 'Исходное изображение',
        'result': 'Результат распознавания',
        'class_col': 'Класс',
        'confidence_col': 'Уверенность',
        'coordinates_col': 'Координаты (x1, y1, x2, y2)',
        'download': 'Скачать результат',
        'no_files': 'Нет файлов',
        'error': 'Произошла ошибка при обработке изображений'
    },
    'kk': {
        'title': 'Теміржолдағы объектілерді анықтау',
        'model': 'Үздік оқытылған модель',
        'conf_threshold': 'Сенімділік шегі (0-1):',
        'iou_threshold': 'IoU шегі (0-1):',
        'drop_files': 'Файлдарды осында сүйреңіз немесе',
        'select_files': 'Файлдарды таңдаңыз',
        'start_detection': 'Анықтауды бастау',
        'clear_all': 'Барлығын тазалау',
        'processing': 'Суреттерді өңдеу...',
        'results': 'Анықтау нәтижелері',
        'original': 'Бастапқы сурет',
        'result': 'Анықтау нәтижесі',
        'class_col': 'Класс',
        'confidence_col': 'Сенімділік',
        'coordinates_col': 'Координаттар (x1, y1, x2, y2)',
        'download': 'Нәтижені жүктеу',
        'no_files': 'Файлдар жоқ',
        'error': 'Суреттерді өңдеу кезінде қате пайда болды'
    },
    'en': {
        'title': 'Railway Object Detection',
        'model': 'Best Trained Model',
        'conf_threshold': 'Confidence Threshold (0-1):',
        'iou_threshold': 'IoU Threshold (0-1):',
        'drop_files': 'Drop files here or',
        'select_files': 'Select Files',
        'start_detection': 'Start Detection',
        'clear_all': 'Clear All',
        'processing': 'Processing images...',
        'results': 'Detection Results',
        'original': 'Original Image',
        'result': 'Detection Result',
        'class_col': 'Class',
        'confidence_col': 'Confidence',
        'coordinates_col': 'Coordinates (x1, y1, x2, y2)',
        'download': 'Download Result',
        'no_files': 'No files',
        'error': 'Error processing images'
    }
}
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB макс размер файла

# Создаем папки если их нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Список доступных моделей
AVAILABLE_MODELS = {
    'best': 'Лучшая обученная модель',
}

# Загружаем модели при старте (ленивая загрузка)
models = {}

def get_model(model_name):
    if model_name not in models:
        if model_name == 'best':
            model_path = 'best.pt'
        else:
            model_path = f'{model_name}.pt'
        models[model_name] = YOLO(model_path)
    return models[model_name]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_detection(model, image_path, conf_threshold=0.25, iou_threshold=0.45):
    results = model(
        image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        save=True,
        save_txt=True
    )
    
    # Собираем статистику детекции
    detections = []
    for r in results[0].boxes:
        det = {
            'class': results[0].names[int(r.cls)],
            'confidence': float(r.conf),
            'bbox': r.xyxy[0].tolist()
        }
        detections.append(det)
    
    return results, detections

@app.route('/', methods=['GET'])
def index():
    lang = request.args.get('lang', session.get('lang', 'kk'))  # Меняем дефолтный язык на 'kk'
    session['lang'] = lang
    return render_template('upload.html', 
                         models=AVAILABLE_MODELS,
                         translations=TRANSLATIONS[lang],
                         current_lang=lang)

@app.route('/detect', methods=['POST'])
def detect():
    if 'files[]' not in request.files:
        return jsonify({'error': 'Нет файлов'}), 400
    
    files = request.files.getlist('files[]')
    model_name = request.form.get('model', 'best')
    conf_threshold = float(request.form.get('conf', 0.25))
    iou_threshold = float(request.form.get('iou', 0.45))
    
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Загружаем модель и запускаем детекцию
            model = get_model(model_name)
            detection_results, detections = process_detection(
                model, filepath, conf_threshold, iou_threshold
            )
            
            # Сохраняем результат
            result_path = os.path.join(RESULTS_FOLDER, filename)
            detection_results[0].save(result_path)
            
            results.append({
                'filename': filename,
                'input_image': f'/uploads/{filename}',
                'result_image': f'/results_web/{filename}',
                'detections': detections
            })
    
    return jsonify(results)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/results_web/<path:filename>')
def result_file(filename):
    return send_file(os.path.join(RESULTS_FOLDER, filename))

@app.route('/clear', methods=['POST'])
def clear_results():
    # Очищаем папки с загруженными файлами и результатами
    for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
        files = glob.glob(os.path.join(folder, '*'))
        for f in files:
            try:
                os.remove(f)
            except:
                pass
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)