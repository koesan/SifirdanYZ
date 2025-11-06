import numpy as np
import pygame
import time
import sys 
import struct # Yerel dosya okuma için
import os 
from sklearn.datasets import fetch_openml # Yedek yükleme için


# --- 0. PYGAME AYARLARI ---
pygame.init()

# Renk Paleti (Canlı ve Kontrastlı)
BLACK = (10, 10, 10)  
WHITE = (240, 240, 240)
ACCENT_GREEN = (0, 255, 100)  # Pozitif aktivasyon/ileri besleme
ACCENT_BLUE = (50, 150, 255)  # Nöron varsayılan
ACCENT_RED = (255, 50, 50)    # Negatif gradyan
ACCENT_YELLOW = (255, 255, 50) # En yüksek tahmin/gradyan

# EKRAN BOYUTU
SCREEN_WIDTH = 1600  
SCREEN_HEIGHT = 1000 
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("MLP Eğitim Simülasyonu (NİHAİ)")
FONT_MAIN = pygame.font.Font(None, 32)
FONT_SMALL = pygame.font.Font(None, 24)
CLOCK = pygame.time.Clock()
FPS = 15 

# --- 1. YEREL VERİ YÜKLEME VE HAZIRLAMA (SİZİN MANTIĞINIZ) ---

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows*cols) 
        return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def load_and_prepare_data(sample_size=500):
    TEST_IMAGES_FILE = "t10k-images.idx3-ubyte"
    TEST_LABELS_FILE = "t10k-labels.idx1-ubyte"
    
    if os.path.exists(TEST_IMAGES_FILE) and os.path.exists(TEST_LABELS_FILE):
        print("Yerel MNIST dosyaları bulunuyor. Veri yükleniyor...")
        try:
            images = load_mnist_images(TEST_IMAGES_FILE)
            labels = load_mnist_labels(TEST_LABELS_FILE)

            # --- SİZİN VERİ BÖLME MANTIĞINIZ UYGULANDI ---
            x_test = images[:1000].T / 255.0
            y_test = labels[:1000]

            x_train = images[1000:].T / 255.0
            y_train = labels[1000:]
            
            print(f"Eğitim verisi yüklendi: {x_train.shape}")
            
            # Görselleştirme için alt küme (Pygame hızı için)
            X_train_sim = x_train[:, :sample_size]
            Y_train_sim = y_train[:sample_size]
            
            return X_train_sim, Y_train_sim, len(np.unique(labels)) # Toplam sınıf sayısı

        except Exception as e:
            print(f"Yerel dosya okuma/bölme hatası: {e}. Program sonlandırılıyor.")
            sys.exit()

    else:
        print("Yerel MNIST dosyaları bulunamadı (t10k-...). Program sonlandırılıyor.")
        sys.exit()


X_train, Y_train, NUM_CLASSES = load_and_prepare_data(sample_size=500)


# --- 2. MLP Sınıfı (Hesaplama Yapısı Korundu ve Stabilize Edildi) ---
class mlp:
    # Orijinal init yapısı korundu
    def __init__(self, x_train, y_train, hidding_layer_size, epoch, lr, num_classes):
        self.x_train = x_train
        self.y_train = y_train
        self.hidding_layer_size = hidding_layer_size
        self.epoch = epoch
        self.lr = lr
        self.output_size = num_classes 
        
        self.weight = []; self.dWeight = []
        self.bias = []; self.dbias = []
        self.linear_output = []; self.activated_output = []
        self.accuracies = []
        self.is_running = True
        self.epoch_counter = 0; self.current_sample_index = 0
        self.layer_sizes = [] 
        
        self.params()

    def relu(self, X): return np.maximum(X, 0)
    def relu_deriv(self, Z): return (Z > 0).astype(float)
    def softmax(self, X):
        e_x = np.exp(X - np.max(X, axis=0, keepdims=True))
        return e_x / e_x.sum(axis=0)
    
    # one_hot metodu, sınıf sayısını self.output_size'dan alır
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, self.output_size)) 
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T
    
    # He Başlatma mantığı korundu
    def params(self):
        input_size = self.x_train.shape[0]
        output_size = self.output_size 
        self.layer_sizes.append(input_size)
        
        H_max = int(2 ** np.floor(np.log2(input_size))) 
        H_hesaplanan = 32 * (2 ** (self.hidding_layer_size - 1))
        H_start = min(H_hesaplanan, H_max) 

        for _ in range(self.hidding_layer_size):
            self.weight.append(np.random.randn(H_start, input_size) * np.sqrt(2 / input_size)) 
            self.bias.append(np.zeros((H_start, 1)))
            self.layer_sizes.append(H_start)
            input_size = H_start
            H_start = max(16, int(H_start / 2)) 

        self.weight.append(np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)) 
        self.bias.append(np.zeros((output_size, 1)))
        self.layer_sizes.append(output_size)
        self.dWeight = [np.zeros_like(w) for w in self.weight]
        self.dbias = [np.zeros_like(b) for b in self.bias]

    # --- TOPLU İLERİ/GERİ BESLEME (Hesaplama) ---
    def forward_prop_batch(self):
        X = self.x_train
        linear_output = []; activated_output = []
        for i in range(len(self.weight)):
            z = self.weight[i].dot(X) + self.bias[i]
            linear_output.append(z)
            X = self.softmax(z) if i == len(self.weight) - 1 else self.relu(z)
            activated_output.append(X)
        self.linear_output = linear_output
        self.activated_output = activated_output
        return X

    def backward_prop_batch(self):
        _, m = self.x_train.shape
        dz_ = self.activated_output[-1] - self.one_hot(self.y_train)

        self.dWeight = [None] * len(self.weight)
        self.dbias = [None] * len(self.bias)

        for i in reversed(range(len(self.weight))):
            a_prev = self.x_train if i == 0 else self.activated_output[i-1]

            self.dWeight[i] = 1 / m * dz_.dot(a_prev.T)
            self.dbias[i] = 1 / m * np.sum(dz_, axis=1, keepdims=True)

            if i != 0:
                dz_ = self.weight[i].T.dot(dz_) * self.relu_deriv(self.linear_output[i-1])

    def update_params_batch(self):
        for i in range(len(self.weight)):
            self.weight[i] = self.weight[i] - self.lr * self.dWeight[i]
            self.bias[i] = self.bias[i] - self.lr * self.dbias[i]

    def get_accuracy(self):
        p = np.argmax(self.activated_output[-1] , 0)
        Y = self.y_train
        return np.mean(p == Y)
        
    # --- GÖRSEL İLERİ BESLEME (Tek Örnek) ---
    def forward_prop_single(self, X_sample, current_sample_index):
        self.linear_output_vis = [] 
        self.activated_output_vis = []
        A = X_sample
        
        self.draw_network(A, 0, 'FORWARD_START', current_sample_index, active_layer_idx=0)
        pygame.display.flip()
        
        for i in range(len(self.weight)):
            Z = self.weight[i].dot(A) + self.bias[i]
            self.linear_output_vis.append(Z)
            A = self.softmax(Z) if i == len(self.weight) - 1 else self.relu(Z)
            self.activated_output_vis.append(A)

            self.draw_network(A, i + 1, f'FORWARD_L{i+1}', current_sample_index, active_layer_idx=i+1, active_prev_weights=i)
            pygame.display.flip()
            
            if i == 0 or i == len(self.weight) - 1:
                time.sleep(0.1) 
        
        return A
        
    # --- GÖRSEL GERİ BESLEME (Tek Örnek) ---
    def backward_prop_single(self, X_sample, Y_sample_array, current_sample_index):
        dz_ = self.activated_output_vis[-1] - self.one_hot(Y_sample_array)

        for i in reversed(range(len(self.weight))):
            self.draw_network(self.activated_output_vis[-1], len(self.weight), f'BACKWARD_L{i+1}', 
                              current_sample_index, active_layer_idx=i+1, active_grad_weights=i, dW_current=self.dWeight[i])
            pygame.display.flip()
            
            if i != 0:
                 # Hata sinyalini görsel olarak geri yayınla
                 dz_ = self.weight[i].T.dot(dz_) * self.relu_deriv(self.linear_output_vis[i-1])


    def train(self):
        for i in range(self.epoch):
            # 1. GERÇEK TOPLU EĞİTİM ADIMI (HESAPLAMA)
            self.forward_prop_batch()
            self.backward_prop_batch()
            self.update_params_batch()
            
            # 2. GÖRSELLEŞTİRME ADIMI
            self.current_sample_index = i % self.x_train.shape[1]
            X_sample = self.x_train[:, self.current_sample_index, None] 
            Y_sample = self.y_train[self.current_sample_index]
            Y_sample_array = np.array([Y_sample]) 
            
            # Tek örnek üzerinde animasyon
            self.forward_prop_single(X_sample, self.current_sample_index)
            self.backward_prop_single(X_sample, Y_sample_array, self.current_sample_index)
            
            self.epoch_counter = i
            
            if i % 10 == 0:
                acc = self.get_accuracy()
                self.accuracies.append(acc)
                print(f"Iteration: {i}, Accuracy: {acc:.4f}")

        pygame.quit()
        print("\nEğitim ve Simülasyon Tamamlandı.")

    # --- 3. Pygame Çizim Fonksiyonları (Görsel Ayarlar) ---

    def get_neuron_pos(self, layer_idx, neuron_idx):
        layer_count = len(self.layer_sizes)
        layer_size = self.layer_sizes[layer_idx]
        total_height_for_neurons = SCREEN_HEIGHT - 150 
        
        x_start_offset = 100 
        x_end_offset = 100   
        x_total_space = SCREEN_WIDTH - x_start_offset - x_end_offset
        x = x_start_offset + layer_idx * (x_total_space / (layer_count - 1)) if layer_count > 1 else SCREEN_WIDTH / 2

        if layer_size > 1:
            y_spacing = (total_height_for_neurons / layer_size)
            
            if layer_size > 500:
                y_spacing = 1.1 
                neuron_radius = 1 
            else:
                neuron_radius = min(15, int(y_spacing * 0.40)) 

            y_center_start = (SCREEN_HEIGHT - (layer_size * y_spacing)) / 2 
            y = y_center_start + neuron_idx * y_spacing
        else:
             neuron_radius = 15
             y = SCREEN_HEIGHT / 2

        return int(x), int(y), neuron_radius

    def draw_network(self, activations_vec, current_layer_idx, phase, sample_idx, active_layer_idx=None, active_prev_weights=None, active_grad_weights=None, dW_current=None):
        SCREEN.fill(BLACK)
        
        # --- Metrikler ---
        text_metrics = FONT_MAIN.render(f"Epoch: {self.epoch_counter}/{self.epoch} | LR: {self.lr} | Örnek: {sample_idx}", True, WHITE)
        SCREEN.blit(text_metrics, (10, 10))
        if self.accuracies:
             text_acc = FONT_MAIN.render(f"Doğruluk: {self.accuracies[-1]:.4f}", True, ACCENT_YELLOW)
             SCREEN.blit(text_acc, (10, 40))
        true_label = self.y_train[sample_idx]
        text_label = FONT_MAIN.render(f"GERÇEK: {true_label}", True, WHITE)
        SCREEN.blit(text_label, (SCREEN_WIDTH - 250, 10))
        text_phase = FONT_SMALL.render(f"FAZ: {phase}", True, WHITE)
        SCREEN.blit(text_phase, (10, SCREEN_HEIGHT - 40))


        # --- Bağlantıları Çizme ---
        for i in range(len(self.weight)):
            prev_layer_size = self.layer_sizes[i]
            curr_layer_size = self.layer_sizes[i+1]
            
            is_forward_active = active_prev_weights == i 
            is_backward_active = active_grad_weights == i
            
            skip_rate = max(1, (prev_layer_size * curr_layer_size) // 2000) 

            for j in range(0, curr_layer_size): 
                for k in range(0, prev_layer_size): 
                    
                    if not is_forward_active and not is_backward_active:
                         if (j * prev_layer_size + k) % skip_rate != 0:
                            continue

                    start_x, start_y, _ = self.get_neuron_pos(i, k)
                    end_x, end_y, _ = self.get_neuron_pos(i+1, j)

                    start_pos = (start_x, start_y)
                    end_pos = (end_x, end_y)

                    weight_val = self.weight[i][j, k]
                    thickness = int(np.clip(abs(weight_val) * 5, 1, 2)) 
                    
                    # 1. TEMEL AĞIRLIK RENGİ
                    normalized_weight = np.clip(weight_val / 0.5, -1, 1) 
                    color_r = int(np.clip(-normalized_weight * 255, 0, 255) * 0.2) 
                    color_g = int(np.clip(normalized_weight * 255, 0, 255) * 0.2)
                    color = (color_r + 5, color_g + 5, 5) 

                    # 2. İLERİ BESLEME VURGUSU
                    if is_forward_active:
                        color = ACCENT_GREEN if weight_val > 0 else ACCENT_RED 
                        thickness = 2 
                        
                    # 3. GERİ BESLEME VURGUSU (Gradyan)
                    elif is_backward_active:
                        delta_w = dW_current[j, k]
                        normalized_delta = np.clip(delta_w * 150, -1, 1) 
                        
                        color_grad_r = int(np.clip(-normalized_delta * 255, 0, 255))
                        color_grad_b = int(np.clip(normalized_delta * 255, 0, 255))
                        color = (color_grad_r, 0, color_grad_b) 
                        thickness = int(np.clip(abs(normalized_delta) * 5, 2, 7))

                    pygame.draw.line(SCREEN, color, start_pos, end_pos, thickness)


        # --- Nöronları Çizme (Aktivasyon Vurgusu) ---
        for i in range(len(self.layer_sizes)):
            layer_size = self.layer_sizes[i]
            
            if i == 0: activations = activations_vec 
            elif i > 0 and i - 1 < len(self.activated_output_vis): activations = self.activated_output_vis[i-1] 
            else: activations = None
            
            max_pred_idx = np.argmax(self.activated_output_vis[-1]) if self.activated_output_vis else -1

            # Katman Adı
            layer_name = "GİRDİ (784)" if i == 0 else ("ÇIKTI (10)" if i == len(self.layer_sizes) - 1 else f"GİZLİ {i} ({layer_size})")
            text_layer = FONT_SMALL.render(layer_name, True, WHITE)
            x_layer, _, _ = self.get_neuron_pos(i, 0)
            SCREEN.blit(text_layer, (x_layer - 40, SCREEN_HEIGHT - 80))
            
            limit = layer_size
            if activations is not None: limit = min(limit, activations.shape[0])
            
            for j in range(0, limit):
                center_x, center_y, neuron_radius = self.get_neuron_pos(i, j)
                center = (center_x, center_y)
                
                neuron_color = ACCENT_BLUE
                
                if activations is not None and activations.shape[1] > 0:
                    activation_val = activations[j, 0]
                    
                    # Nöron Rengi (Aktif Katman Vurgusu)
                    if active_layer_idx == i:
                        normalized_activation = np.clip(activation_val * 4, 0, 1) 
                        color_intensity = int(normalized_activation * 255) 
                        neuron_color = (color_intensity, color_intensity, color_intensity) 
                    else:
                        normalized_activation = np.clip(activation_val, 0, 1)
                        color_intensity = int(normalized_activation * 150)
                        neuron_color = (0, color_intensity, color_intensity) 
                    
                    # Çıkış katmanı vurgusu
                    if i == len(self.layer_sizes) - 1:
                        if j == max_pred_idx:
                            neuron_color = ACCENT_YELLOW
                            
                        if self.activated_output_vis:
                            pred_prob = self.activated_output_vis[-1][j, 0]
                            text_prob = FONT_SMALL.render(f"{j}: {pred_prob:.2f}", True, WHITE)
                            SCREEN.blit(text_prob, (center_x + neuron_radius + 5, center_y - 10))

                # Nöronu çiz
                pygame.draw.circle(SCREEN, neuron_color, center, neuron_radius)
                pygame.draw.circle(SCREEN, WHITE, center, neuron_radius, 1) 


# --- 5. Uygulamayı Başlatma ---

# Deneyin: Hidding_layer_size=4, lr=0.1
a = mlp(X_train, Y_train, hidding_layer_size=3, epoch=250, lr=0.1, num_classes=NUM_CLASSES) 
a.train()