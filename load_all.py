import os
import cv2
import numpy as np
from tqdm import tqdm

def load_images_from_folder(folder_path, target_size=(640, 640), normalize=False): # çok yer kaplamasın diye 224x224 yaptım
    images = []
    file_paths = []
    
    for filename in tqdm(os.listdir(folder_path), desc=f"Loading {os.path.basename(folder_path)}"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')): #resim dosyalarını alıyor 
            file_path = os.path.join(folder_path, filename)
            
            img = cv2.imread(file_path)
            if img is not None:  # başarılı ile okunduysa dönüşüm yapıp 224x224 boyutuna resize yapıyor
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                
                if normalize:
                    img = img.astype(np.float32) / 255.0   #normalleştirmek şart değil ama deep learning ağı için daha stabil şekilde türev değerleri güncellenir
                    # Hem de Deep learning modelleri 0 ile 1 arasındaki değerler ile daha iyi performans gösterirler. Fakat şu anlık burda bir normalleştirme yapmıyoruz.
                
                images.append(img)
                file_paths.append(file_path)
    
    return np.array(images), file_paths  # np.array önemli

# Veri yollarını tanımla
base_path = 'assets/Gopro/Gopro'  # temel yol bu bunun üstünden gidelim
paths = {
    'x_train': os.path.join(base_path, 'train/blur'),
    'y_train': os.path.join(base_path, 'train/sharp'),
    'x_test': os.path.join(base_path, 'test/blur'),
    'y_test': os.path.join(base_path, 'test/sharp')
}

# Verileri yükle
data = {}
for key, path in paths.items():     # x_train, x_test, y_train, y_test olarak ayrı ayrı almak istiyorum
    print(f"\nYükleniyor: {key}")
    images, _ = load_images_from_folder(path, normalize=False)  # normalize yok 
    data[key] = images
    print(f"{key} shape: {images.shape}")

# NumPy dosyası olarak kaydet
save_path = 'gopro_dataset.npz'         
np.savez_compressed(save_path, 
                   x_train=data['x_train'],
                   y_train=data['y_train'],
                   x_test=data['x_test'],
                   y_test=data['y_test'])

print(f"\nVeriler kaydedildi: {save_path}")

# Dosyayı test amaçlı yükle ve kontrol et
loaded_data = np.load(save_path)
for key in loaded_data.files:
    print(f"\n{key} shape:", loaded_data[key].shape)  



# YÜKLEDİĞİM VERİLERİDE BU ŞEKİLDE HER SEFERİNDE KULLANABİLİYORUM

# neden .npz tipinde dosyayı kaydettik çünkü .npz Numpy'a özel bir dizi sıkıştırma dosya tipidir, .npz hızlıdır boyutlar korunur
# ve tek kötü tarafı sadece numpy ile çalışmasıdır
data = np.load('gopro_dataset.npz')
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

print(f"x_train boyutu: {x_train.shape}")  # (2103, 640, 640, 3) #(veri_sayısı, height, width, channels)
print(f"x_test boyutu: {x_test.shape}")    # (1111, 640, 640, 3) #channels = 3 çünkü veri bgr bir resim, eğer grayscale olsaydı 1 olurdu
print(f"y_train boyutu: {y_train.shape}")  # (2103, 640, 640, 3)
print(f"y_test boyutu: {y_test.shape}")    # (1111, 640, 640, 3)


# İLERİDE MİNİ-BATCHLER KULLANILARAK DAHA VERİMLİ BİR TRAİNİNG SÜRECİNE ULAŞILABİLİR
