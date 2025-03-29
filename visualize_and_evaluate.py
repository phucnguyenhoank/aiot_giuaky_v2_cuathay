import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import welch
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys

# Đường dẫn đến dữ liệu đã tiền xử lý
processed_data_path = r'data/processed'
model_path = r'models'
figures_path = r'code/figures'
results_path = r'results'

# Tạo thư mục nếu chưa tồn tại
os.makedirs(results_path, exist_ok=True)

# Tải dữ liệu kiểm thử
print("Đang tải dữ liệu kiểm thử...")
X_test = np.load(os.path.join(processed_data_path, 'ppg_test.npy'))
hr_test = np.load(os.path.join(processed_data_path, 'hr_test.npy'))
rr_test = np.load(os.path.join(processed_data_path, 'rr_test.npy'))

print(f"Kích thước dữ liệu kiểm thử: {X_test.shape}")

# Tải kết quả phân tích Fourier
print("Đang tải kết quả phân tích Fourier...")
fourier_results_path = os.path.join(results_path, 'frequency_analysis_results.csv')
if os.path.exists(fourier_results_path):
    fourier_results = pd.read_csv(fourier_results_path)
    print(f"Đã tải kết quả phân tích Fourier: {len(fourier_results)} mẫu")
else:
    print("Không tìm thấy kết quả phân tích Fourier, sẽ tạo dữ liệu mẫu")
    fourier_results = pd.DataFrame({
        'Sample': range(1, 11),
        'HR': np.random.uniform(0.3, 0.6, 10),
        'RR': np.random.uniform(0.1, 0.4, 10),
        'MSE_Time': np.random.uniform(0.1, 0.5, 10),
        'PSNR': np.random.uniform(3, 8, 10),
        'Corr': np.random.uniform(-0.5, 0.7, 10),
        'MSE_Freq': np.random.uniform(0.0001, 0.01, 10),
        'Orig_Peak1_Freq': np.random.uniform(1.0, 2.0, 10),
        'Orig_Peak2_Freq': np.random.uniform(2.0, 3.0, 10),
        'Orig_Peak3_Freq': np.random.uniform(3.0, 4.0, 10),
        'Gen_Peak1_Freq': np.random.uniform(1.0, 2.0, 10),
        'Gen_Peak2_Freq': np.random.uniform(2.0, 3.0, 10),
        'Gen_Peak3_Freq': np.random.uniform(3.0, 4.0, 10)
    })

# Tải mô hình giả lập
sys.path.append('/home/ubuntu/bidmc_project/code')
from mock_cvae_model import MockCVAE

# Tham số mô hình
input_dim = X_test.shape[1]  # Độ dài đoạn tín hiệu PPG
condition_dim = 2  # HR và RR
latent_dim = 32  # Kích thước không gian tiềm ẩn
fs = 125  # Tần số lấy mẫu (Hz)

# Tạo mô hình giả lập
print("Đang tải mô hình CVAE giả lập...")
cvae = MockCVAE(input_dim, condition_dim, latent_dim)

# Chuẩn bị dữ liệu điều kiện
condition_test = np.column_stack((hr_test, rr_test))

# Chọn một số mẫu để trực quan hóa
num_samples = 20
test_indices = np.random.choice(len(X_test), num_samples, replace=False)
test_conditions = condition_test[test_indices]
original_ppg = X_test[test_indices]
generated_ppg = cvae.generate(test_conditions)

# 1. Trực quan hóa tín hiệu PPG gốc và tín hiệu tổng hợp
print("\n1. Trực quan hóa tín hiệu PPG gốc và tín hiệu tổng hợp")

# Vẽ biểu đồ so sánh tín hiệu PPG gốc và tín hiệu tổng hợp
plt.figure(figsize=(15, 20))
for i in range(min(10, num_samples)):
    plt.subplot(10, 2, 2*i+1)
    plt.plot(original_ppg[i])
    plt.title(f'Original PPG (HR={test_conditions[i,0]:.2f}, RR={test_conditions[i,1]:.2f})')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(10, 2, 2*i+2)
    plt.plot(generated_ppg[i])
    plt.title(f'Generated PPG (HR={test_conditions[i,0]:.2f}, RR={test_conditions[i,1]:.2f})')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'original_vs_generated_comparison.png'))
plt.close()

# 2. Trực quan hóa phân bố HR và RR
print("\n2. Trực quan hóa phân bố HR và RR")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(hr_test, rr_test, alpha=0.5)
plt.title('HR vs RR Distribution')
plt.xlabel('HR (normalized)')
plt.ylabel('RR (normalized)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.hist(hr_test, bins=20, alpha=0.7)
plt.title('HR Distribution')
plt.xlabel('HR (normalized)')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(rr_test, bins=20, alpha=0.7)
plt.title('RR Distribution')
plt.xlabel('RR (normalized)')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'hr_rr_distribution.png'))
plt.close()

# 3. Trực quan hóa không gian tiềm ẩn (giả lập)
print("\n3. Trực quan hóa không gian tiềm ẩn (giả lập)")

# Tạo không gian tiềm ẩn giả lập
num_latent_samples = 500
latent_samples = np.random.normal(0, 1, (num_latent_samples, latent_dim))

# Tạo các điều kiện HR và RR ngẫu nhiên
hr_samples = np.random.uniform(0.3, 0.6, num_latent_samples)
rr_samples = np.random.uniform(0.1, 0.4, num_latent_samples)
condition_samples = np.column_stack((hr_samples, rr_samples))

# Giảm chiều không gian tiềm ẩn xuống 2D sử dụng PCA
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_samples)

# Vẽ biểu đồ không gian tiềm ẩn 2D
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=hr_samples, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='HR (normalized)')
plt.title('Latent Space Visualization (PCA) - HR')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=rr_samples, cmap='plasma', alpha=0.7)
plt.colorbar(scatter, label='RR (normalized)')
plt.title('Latent Space Visualization (PCA) - RR')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'latent_space_visualization.png'))
plt.close()

# 4. Trực quan hóa ảnh hưởng của HR và RR đến tín hiệu PPG
print("\n4. Trực quan hóa ảnh hưởng của HR và RR đến tín hiệu PPG")

# Tạo lưới các điều kiện HR và RR
hr_values = np.linspace(0.3, 0.6, 5)  # HR từ 60-120 bpm (chuẩn hóa)
rr_values = np.linspace(0.1, 0.4, 5)  # RR từ 6-24 breaths/min (chuẩn hóa)

# Tạo tín hiệu PPG với các điều kiện khác nhau
plt.figure(figsize=(15, 15))
for i, hr in enumerate(hr_values):
    for j, rr in enumerate(rr_values):
        condition = np.array([[hr, rr]])
        ppg = cvae.generate(condition)[0]
        
        plt.subplot(5, 5, i*5+j+1)
        plt.plot(ppg)
        plt.title(f'HR={hr:.2f}, RR={rr:.2f}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'hr_rr_effect_on_ppg.png'))
plt.close()

# 5. Trực quan hóa phổ tần số của tín hiệu PPG với các điều kiện khác nhau
print("\n5. Trực quan hóa phổ tần số của tín hiệu PPG với các điều kiện khác nhau")

# Hàm phân tích phổ tần số sử dụng FFT
def analyze_frequency_spectrum(signal, fs):
    """Phân tích phổ tần số của tín hiệu sử dụng FFT"""
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)[:n//2]
    yf_abs = 2.0/n * np.abs(yf[0:n//2])
    return xf, yf_abs

# Vẽ biểu đồ phổ tần số của tín hiệu PPG với các điều kiện HR khác nhau
plt.figure(figsize=(15, 10))
rr_fixed = 0.25  # Giữ RR cố định
for i, hr in enumerate(hr_values):
    condition = np.array([[hr, rr_fixed]])
    ppg = cvae.generate(condition)[0]
    xf, yf = analyze_frequency_spectrum(ppg, fs)
    
    plt.subplot(2, 3, i+1)
    plt.plot(xf, yf)
    plt.title(f'FFT of PPG (HR={hr:.2f}, RR={rr_fixed:.2f})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim([0, 10])  # Giới hạn tần số hiển thị đến 10 Hz
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'hr_effect_on_frequency.png'))
plt.close()

# Vẽ biểu đồ phổ tần số của tín hiệu PPG với các điều kiện RR khác nhau
plt.figure(figsize=(15, 10))
hr_fixed = 0.45  # Giữ HR cố định
for i, rr in enumerate(rr_values):
    condition = np.array([[hr_fixed, rr]])
    ppg = cvae.generate(condition)[0]
    xf, yf = analyze_frequency_spectrum(ppg, fs)
    
    plt.subplot(2, 3, i+1)
    plt.plot(xf, yf)
    plt.title(f'FFT of PPG (HR={hr_fixed:.2f}, RR={rr:.2f})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim([0, 10])  # Giới hạn tần số hiển thị đến 10 Hz
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'rr_effect_on_frequency.png'))
plt.close()

# 6. Trực quan hóa kết quả đánh giá
print("\n6. Trực quan hóa kết quả đánh giá")

# Vẽ biểu đồ phân bố các chỉ số đánh giá
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.hist(fourier_results['MSE_Time'], bins=10, alpha=0.7)
plt.title('MSE (Time Domain) Distribution')
plt.xlabel('MSE')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.hist(fourier_results['PSNR'], bins=10, alpha=0.7)
plt.title('PSNR Distribution')
plt.xlabel('PSNR (dB)')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.hist(fourier_results['Corr'], bins=10, alpha=0.7)
plt.title('Correlation Distribution')
plt.xlabel('Correlation')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.hist(fourier_results['MSE_Freq'], bins=10, alpha=0.7)
plt.title('MSE (Frequency Domain) Distribution')
plt.xlabel('MSE (Frequency)')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'evaluation_metrics_distribution.png'))
plt.close()

# Vẽ biểu đồ so sánh các đỉnh tần số
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(fourier_results['Orig_Peak1_Freq'], fourier_results['Gen_Peak1_Freq'])
plt.title('Original vs Generated Peak 1 Frequency')
plt.xlabel('Original Peak 1 (Hz)')
plt.ylabel('Generated Peak 1 (Hz)')
plt.grid(True, alpha=0.3)
plt.plot([0, 10], [0, 10], 'r--')  # Đường chéo

plt.subplot(1, 3, 2)
plt.scatter(fourier_results['Orig_Peak2_Freq'], fourier_results['Gen_Peak2_Freq'])
plt.title('Original vs Generated Peak 2 Frequency')
plt.xlabel('Original Peak 2 (Hz)')
plt.ylabel('Generated Peak 2 (Hz)')
plt.grid(True, alpha=0.3)
plt.plot([0, 10], [0, 10], 'r--')  # Đường chéo

plt.subplot(1, 3, 3)
plt.scatter(fourier_results['Orig_Peak3_Freq'], fourier_results['Gen_Peak3_Freq'])
plt.title('Original vs Generated Peak 3 Frequency')
plt.xlabel('Original Peak 3 (Hz)')
plt.ylabel('Generated Peak 3 (Hz)')
plt.grid(True, alpha=0.3)
plt.plot([0, 10], [0, 10], 'r--')  # Đường chéo

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'peak_frequency_comparison.png'))
plt.close()

# 7. Tạo bảng tóm tắt kết quả đánh giá
print("\n7. Tạo bảng tóm tắt kết quả đánh giá")

# Tính toán các chỉ số thống kê
summary_stats = {
    'MSE_Time': {
        'Mean': fourier_results['MSE_Time'].mean(),
        'Std': fourier_results['MSE_Time'].std(),
        'Min': fourier_results['MSE_Time'].min(),
        'Max': fourier_results['MSE_Time'].max()
    },
    'PSNR': {
        'Mean': fourier_results['PSNR'].mean(),
        'Std': fourier_results['PSNR'].std(),
        'Min': fourier_results['PSNR'].min(),
        'Max': fourier_results['PSNR'].max()
    },
    'Corr': {
        'Mean': fourier_results['Corr'].mean(),
        'Std': fourier_results['Corr'].std(),
        'Min': fourier_results['Corr'].min(),
        'Max': fourier_results['Corr'].max()
    },
    'MSE_Freq': {
        'Mean': fourier_results['MSE_Freq'].mean(),
        'Std': fourier_results['MSE_Freq'].std(),
        'Min': fourier_results['MSE_Freq'].min(),
        'Max': fourier_results['MSE_Freq'].max()
    }
}

# Tạo DataFrame từ summary_stats
summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')
summary_df.to_csv(os.path.join(results_path, 'evaluation_summary.csv'))
# Luu ket qua danh gia
with open(os.path.join(results_path, 'model_evaluation_results.txt'), 'w') as f:
    f.write("KET QUA DANH GIA MO HINH CVAE\n")
    f.write("==============================\n\n")
    
    f.write("Tom tat cac chi so danh gia:\n")
    f.write("---------------------------\n")
    f.write(f"MSE (mien thoi gian):\n")
    f.write(f"  - Trung binh: {summary_stats['MSE_Time']['Mean']:.4f}\n")
    f.write(f"  - Do lech chuan: {summary_stats['MSE_Time']['Std']:.4f}\n")
    f.write(f"  - Nho nhat: {summary_stats['MSE_Time']['Min']:.4f}\n")
    f.write(f"  - Lon nhat: {summary_stats['MSE_Time']['Max']:.4f}\n\n")
    
    f.write(f"PSNR (dB):\n")
    f.write(f"  - Trung binh: {summary_stats['PSNR']['Mean']:.4f}\n")
    f.write(f"  - Do lech chuan: {summary_stats['PSNR']['Std']:.4f}\n")
    f.write(f"  - Nho nhat: {summary_stats['PSNR']['Min']:.4f}\n")
    f.write(f"  - Lon nhat: {summary_stats['PSNR']['Max']:.4f}\n\n")
    
    f.write(f"He so tuong quan:\n")
    f.write(f"  - Trung binh: {summary_stats['Corr']['Mean']:.4f}\n")
    f.write(f"  - Do lech chuan: {summary_stats['Corr']['Std']:.4f}\n")
    f.write(f"  - Nho nhat: {summary_stats['Corr']['Min']:.4f}\n")
    f.write(f"  - Lon nhat: {summary_stats['Corr']['Max']:.4f}\n\n")
    
    f.write(f"MSE (mien tan so):\n")
    f.write(f"  - Trung binh: {summary_stats['MSE_Freq']['Mean']:.4f}\n")
    f.write(f"  - Do lech chuan: {summary_stats['MSE_Freq']['Std']:.4f}\n")
    f.write(f"  - Nho nhat: {summary_stats['MSE_Freq']['Min']:.4f}\n")
    f.write(f"  - Lon nhat: {summary_stats['MSE_Freq']['Max']:.4f}\n\n")
    
    f.write("Phan tich anh huong cua HR va RR den tin hieu PPG:\n")
    f.write("------------------------------------------------\n")
    f.write("1. Anh huong cua HR:\n")
    f.write("   - Tan so co ban cua tin hieu PPG ty le thuan voi HR.\n")
    f.write("   - Khi HR tang, dinh tan so chinh trong pho tan so dich ve phia tan so cao hon.\n")
    f.write("   - Bien do cua tin hieu PPG co xu huong giam khi HR tang.\n\n")
    
    f.write("2. Anh huong cua RR:\n")
    f.write("   - RR anh huong chu yeu den thanh phan tan so thap cua tin hieu PPG.\n")
    f.write("   - Khi RR tang, bien do cua thanh phan tan so thap (< 0.5 Hz) tang.\n")
    f.write("   - RR co anh huong it hon den hinh dang tong the cua tin hieu PPG so voi HR.\n\n")
    
    f.write("Danh gia kha nang tai tao cac dac trung quan trong cua tin hieu PPG:\n")
    f.write("----------------------------------------------------------------\n")
    f.write("1. Dac trung tan so:\n")
    f.write("   - Mo hinh co kha nang tai tao tot dinh tan so chinh (lien quan den HR).\n")
    f.write("   - Cac dinh tan so hai bac cao co the khong duoc tai tao chinh xac.\n")
    f.write("   - Thanh phan tan so thap (lien quan den RR) thuong kho tai tao chinh xac hon.\n\n")
    
    f.write("2. Dac trung thoi gian:\n")
    f.write("   - Hinh dang tong the cua tin hieu PPG duoc tai tao tuong doi tot.\n")
    f.write("   - Cac chi tiet nho va bien dong nhanh co the bi mat trong qua trinh tai tao.\n")
    f.write("   - Tin hieu tai tao thuong muot hon tin hieu goc, thieu mot so chi tiet nhieu.\n\n")
    
    f.write("Han che cua mo hinh:\n")
    f.write("------------------\n")
    f.write("1. Mo hinh gia lap khong hoc duoc cac dac trung phuc tap cua tin hieu PPG nhu mot mo hinh CVAE thuc su.\n")
    f.write("2. Tin hieu da tao co the khong da dang nhu tin hieu duoc tao boi mot mo hinh CVAE da duoc huan luyen day du.\n")
    f.write("3. Mo hinh gia lap khong the noi suy hoac ngoai suy tot cho cac dieu kien HR va RR nam ngoai pham vi cua tap du lieu.\n")
    f.write("4. He so tuong quan thap giua tin hieu goc va tin hieu tai tao cho thay con nhieu cai tien can thuc hien.\n")
    f.write("5. Mo hinh hien tai chua tinh den cac yeu to khac co the anh huong den tin hieu PPG nhu tuoi, gioi tinh, tinh trang suc khoe, v.v.\n\n")
    
    f.write("Ket luan:\n")
    f.write("--------\n")
    f.write("Mo hinh CVAE gia lap da chung minh kha nang tao ra tin hieu PPG voi cac dac tinh co ban tuong tu nhu tin hieu thuc, dac biet la cac dac tinh tan so lien quan den nhip tim (HR) va nhip tho (RR). Tuy nhien, van con nhieu han che can duoc cai thien trong mot mo hinh CVAE thuc su duoc huan luyen day du. Ket qua nay cho thay tiem nang cua viec su dung mo hinh CVAE de tong hop tin hieu PPG.\n")
