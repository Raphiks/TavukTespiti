import cv2
from ultralytics import YOLO
from flask import Flask, Response, jsonify, render_template
import threading

# Flask uygulamasını başlat
app = Flask(__name__)

# YOLO modelini yükle
model = YOLO("yolov8n.pt")  # Eğitilmiş modelinizi buraya yükleyin

# Kamerayı başlat
cap = cv2.VideoCapture(0)  # 0, varsayılan kamera için

# Kameradan görüntü alıp işleyen fonksiyon
def generate_frames():
    while True:
        ret, frame = cap.read()  # Kameradan bir kare yakala
        if not ret:
            break

        # YOLO ile nesne tespiti yap
        results = model(frame)

        # Tespit edilen nesneleri görüntüye çiz
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box koordinatları
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Kutu çiz
                cv2.putText(frame, f"Tavuk {box.conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Görüntüyü JPEG formatına dönüştür
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Görüntüyü akışa ekle
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Canlı kamera akışı için route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Ana sayfa için route
@app.route('/')
def index():
    return render_template('index.html')  # Basit bir HTML sayfası döndür

# Tavuk sayısını döndüren API
@app.route('/tavuk_sayisi', methods=['GET'])
def tavuk_sayisi():
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Kamera bağlantısı kurulamadı"}), 500

    # YOLO ile nesne tespiti yap
    results = model(frame)

    # Tespit edilen tavukları say
    tavuk_sayisi = 0
    for result in results:
        tavuk_sayisi += len(result.boxes)

    return jsonify({"tavuk_sayisi": tavuk_sayisi})

# Flask uygulamasını başlat
if __name__ == '__main__':
    # Flask'ı ayrı bir thread'de çalıştır
    flask_thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 5000})
    flask_thread.daemon = True
    flask_thread.start()

    # Ana döngüyü çalıştır
    try:
        while True:
            pass  # Programın sürekli çalışmasını sağla
    except KeyboardInterrupt:
        print("Program sonlandırılıyor...")
    finally:
        # Kaynakları serbest bırak
        cap.release()
        cv2.destroyAllWindows()