from ultralytics import YOLO
import os

# Papka va fayllarni tekshirish uchun diagnostika
print("Current Directory:", os.getcwd())
print("Train Images Exists:", os.path.exists(r"C:\Users\13\Desktop\milk\train\images"))
print("Valid Images Exists:", os.path.exists(r"C:\Users\13\Desktop\milk\valid\images"))

# YOLO modelni o'qitish funksiyasi
def main():
    # YOLO modelni yuklash
    model = YOLO("yolo11n.pt")  # Trained yoki yangi model

    # O'qitish
    results = model.train(
        data="data.yaml",  # YAML faylning to'g'ri yo'li
        epochs=100,        # Epochlar soni
        imgsz=640,         # Tasvir o'lchami
        batch=16,          # Batch hajmi
        workers=8,         # Parallel jarayonlar soni
        device="cpu",      # CPU yoki GPU
        pretrained=True,   # Pretrained modeldan foydalanish
        patience=50,       # Early stopping patience
        save=True,         # Modelni saqlash
        save_period=10,    # Har 10 epochda saqlash
        single_cls=True    # Faqat bitta klass uchun
    )
    print("Model training completed.")

if __name__ == "__main__":
    main()
