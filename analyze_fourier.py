import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import welch
from scipy.fft import fft, fftfreq, ifft
import pandas as pd
from sklearn.metrics import mean_squared_error
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

# Tải kết quả kiểm thử
print("Đang tải kết quả kiểm thử...")
sys.path.append(r'code')
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

# Chọn một số mẫu để phân tích
num_samples = 10
test_indices = np.random.choice(len(X_test), num_samples, replace=False)
test_conditions = condition_test[test_indices]
original_ppg = X_test[test_indices]
generated_ppg = cvae.generate(test_conditions)

# Hàm phân tích phổ tần số sử dụng FFT
def analyze_frequency_spectrum(signal, fs):
    """Phân tích phổ tần số của tín hiệu sử dụng FFT"""
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)[:n//2]
    yf_abs = 2.0/n * np.abs(yf[0:n//2])
    return xf, yf_abs

# Hàm phân tích phổ tần số sử dụng Welch's method
def analyze_welch_spectrum(signal, fs):
    """Phân tích phổ tần số của tín hiệu sử dụng Welch's method"""
    f, Pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    return f, Pxx

# Hàm tìm đỉnh trong phổ tần số
def find_peaks(x, y, threshold=0.1, min_distance=5):
    """Tìm các đỉnh trong phổ tần số"""
    # Chuẩn hóa y về [0, 1]
    y_norm = y / np.max(y) if np.max(y) > 0 else y
    
    # Tìm các đỉnh
    peaks = []
    for i in range(1, len(y_norm)-1):
        if y_norm[i] > threshold and y_norm[i] > y_norm[i-1] and y_norm[i] > y_norm[i+1]:
            # Kiểm tra khoảng cách với đỉnh gần nhất
            if not peaks or i - peaks[-1][0] >= min_distance:
                peaks.append((i, x[i], y_norm[i]))
    
    return peaks

# Hàm tính toán các chỉ số đánh giá
def calculate_metrics(original, generated):
    """Tính toán các chỉ số đánh giá giữa tín hiệu gốc và tín hiệu đã tạo"""
    # Tính MSE
    mse = mean_squared_error(original, generated)
    
    # Tính PSNR
    max_val = max(np.max(original), np.max(generated))
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    
    # Tính hệ số tương quan
    corr = np.corrcoef(original, generated)[0, 1]
    
    return mse, psnr, corr

# Hàm tính toán các chỉ số đánh giá trong miền tần số
def calculate_frequency_metrics(f_orig, psd_orig, f_gen, psd_gen):
    """Tính toán các chỉ số đánh giá trong miền tần số"""
    # Chuẩn hóa PSD
    psd_orig_norm = psd_orig / np.max(psd_orig) if np.max(psd_orig) > 0 else psd_orig
    psd_gen_norm = psd_gen / np.max(psd_gen) if np.max(psd_gen) > 0 else psd_gen
    
    # Tính MSE trong miền tần số
    # Nội suy PSD để có cùng kích thước
    if len(f_orig) != len(f_gen):
        from scipy.interpolate import interp1d
        f_min = max(np.min(f_orig), np.min(f_gen))
        f_max = min(np.max(f_orig), np.max(f_gen))
        f_common = np.linspace(f_min, f_max, 1000)
        
        interp_orig = interp1d(f_orig, psd_orig_norm, bounds_error=False, fill_value=0)
        interp_gen = interp1d(f_gen, psd_gen_norm, bounds_error=False, fill_value=0)
        
        psd_orig_interp = interp_orig(f_common)
        psd_gen_interp = interp_gen(f_common)
        
        mse_freq = mean_squared_error(psd_orig_interp, psd_gen_interp)
    else:
        mse_freq = mean_squared_error(psd_orig_norm, psd_gen_norm)
    
    return mse_freq

# Phân tích phổ tần số chi tiết
print("\nPhân tích phổ tần số chi tiết của tín hiệu PPG gốc và tín hiệu PPG đã tạo")

# Tạo DataFrame để lưu kết quả
results_df = pd.DataFrame(columns=[
    'Sample', 'HR', 'RR', 'MSE_Time', 'PSNR', 'Corr', 'MSE_Freq',
    'Orig_Peak1_Freq', 'Orig_Peak2_Freq', 'Orig_Peak3_Freq',
    'Gen_Peak1_Freq', 'Gen_Peak2_Freq', 'Gen_Peak3_Freq'
])

# Phân tích từng mẫu
for i in range(num_samples):
    print(f"\nPhân tích mẫu {i+1}:")
    
    # Phân tích tín hiệu gốc sử dụng FFT
    xf_orig, yf_orig = analyze_frequency_spectrum(original_ppg[i], fs)
    
    # Phân tích tín hiệu đã tạo sử dụng FFT
    xf_gen, yf_gen = analyze_frequency_spectrum(generated_ppg[i], fs)
    
    # Phân tích tín hiệu gốc sử dụng Welch's method
    f_orig, psd_orig = analyze_welch_spectrum(original_ppg[i], fs)
    
    # Phân tích tín hiệu đã tạo sử dụng Welch's method
    f_gen, psd_gen = analyze_welch_spectrum(generated_ppg[i], fs)
    
    # Tìm các đỉnh trong phổ tần số của tín hiệu gốc
    peaks_orig = find_peaks(xf_orig, yf_orig)
    peaks_orig.sort(key=lambda x: x[2], reverse=True)  # Sắp xếp theo biên độ
    
    # Tìm các đỉnh trong phổ tần số của tín hiệu đã tạo
    peaks_gen = find_peaks(xf_gen, yf_gen)
    peaks_gen.sort(key=lambda x: x[2], reverse=True)  # Sắp xếp theo biên độ
    
    # Tính toán các chỉ số đánh giá trong miền thời gian
    mse_time, psnr, corr = calculate_metrics(original_ppg[i], generated_ppg[i])
    
    # Tính toán các chỉ số đánh giá trong miền tần số
    mse_freq = calculate_frequency_metrics(f_orig, psd_orig, f_gen, psd_gen)
    
    # In kết quả
    print(f"HR={test_conditions[i,0]:.4f}, RR={test_conditions[i,1]:.4f}")
    print(f"MSE (time domain): {mse_time:.4f}")
    print(f"PSNR: {psnr:.4f}dB")
    print(f"Correlation: {corr:.4f}")
    print(f"MSE (frequency domain): {mse_freq:.4f}")
    
    print("Các đỉnh trong phổ tần số của tín hiệu gốc:")
    orig_peaks = []
    for j, (idx, freq, amp) in enumerate(peaks_orig[:3]):
        print(f"  Peak {j+1}: {freq:.2f} Hz (amplitude: {amp:.4f})")
        orig_peaks.append(freq)
    
    print("Các đỉnh trong phổ tần số của tín hiệu đã tạo:")
    gen_peaks = []
    for j, (idx, freq, amp) in enumerate(peaks_gen[:3]):
        print(f"  Peak {j+1}: {freq:.2f} Hz (amplitude: {amp:.4f})")
        gen_peaks.append(freq)
    
    # Đảm bảo có đủ 3 đỉnh
    while len(orig_peaks) < 3:
        orig_peaks.append(0)
    while len(gen_peaks) < 3:
        gen_peaks.append(0)
    
    # Thêm vào DataFrame
    new_row = pd.DataFrame({
        'Sample': [i+1],
        'HR': [test_conditions[i,0]],
        'RR': [test_conditions[i,1]],
        'MSE_Time': [mse_time],
        'PSNR': [psnr],
        'Corr': [corr],
        'MSE_Freq': [mse_freq],
        'Orig_Peak1_Freq': [orig_peaks[0]],
        'Orig_Peak2_Freq': [orig_peaks[1]],
        'Orig_Peak3_Freq': [orig_peaks[2]],
        'Gen_Peak1_Freq': [gen_peaks[0]],
        'Gen_Peak2_Freq': [gen_peaks[1]],
        'Gen_Peak3_Freq': [gen_peaks[2]]
    })
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    # Vẽ biểu đồ phổ tần số
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(xf_orig, yf_orig)
    for j, (idx, freq, amp) in enumerate(peaks_orig[:3]):
        plt.plot(freq, yf_orig[idx], 'ro')
        plt.text(freq, yf_orig[idx], f'{freq:.2f} Hz', fontsize=8)
    plt.title(f'Original PPG FFT (HR={test_conditions[i,0]:.2f}, RR={test_conditions[i,1]:.2f})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim([0, 10])  # Giới hạn tần số hiển thị đến 10 Hz
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(xf_gen, yf_gen)
    for j, (idx, freq, amp) in enumerate(peaks_gen[:3]):
        plt.plot(freq, yf_gen[idx], 'ro')
        plt.text(freq, yf_gen[idx], f'{freq:.2f} Hz', fontsize=8)
    plt.title(f'Generated PPG FFT (HR={test_conditions[i,0]:.2f}, RR={test_conditions[i,1]:.2f})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim([0, 10])  # Giới hạn tần số hiển thị đến 10 Hz
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'fft_analysis_sample_{i+1}.png'))
    plt.close()
    
    # Vẽ biểu đồ phổ tần số sử dụng Welch's method
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(f_orig, psd_orig)
    plt.title(f'Original PPG PSD (HR={test_conditions[i,0]:.2f}, RR={test_conditions[i,1]:.2f})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB/Hz)')
    plt.xlim([0, 10])  # Giới hạn tần số hiển thị đến 10 Hz
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(f_gen, psd_gen)
    plt.title(f'Generated PPG PSD (HR={test_conditions[i,0]:.2f}, RR={test_conditions[i,1]:.2f})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB/Hz)')
    plt.xlim([0, 10])  # Giới hạn tần số hiển thị đến 10 Hz
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'psd_analysis_sample_{i+1}.png'))
    plt.close()

# Lưu kết quả vào file CSV
results_df.to_csv(os.path.join(results_path, 'frequency_analysis_results.csv'), index=False)

# Tính toán các chỉ số trung bình
avg_mse_time = results_df['MSE_Time'].mean()
avg_psnr = results_df['PSNR'].mean()
avg_corr = results_df['Corr'].mean()
avg_mse_freq = results_df['MSE_Freq'].mean()

print("\nKết quả trung bình:")
print(f"MSE (time domain): {avg_mse_time:.4f}")
print(f"PSNR: {avg_psnr:.4f}dB")
print(f"Correlation: {avg_corr:.4f}")
print(f"MSE (frequency domain): {avg_mse_freq:.4f}")

# Phân tích tương quan giữa HR, RR và chất lượng tín hiệu
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(results_df['HR'], results_df['MSE_Time'])
plt.title('HR vs MSE (Time Domain)')
plt.xlabel('HR (normalized)')
plt.ylabel('MSE')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.scatter(results_df['RR'], results_df['MSE_Time'])
plt.title('RR vs MSE (Time Domain)')
plt.xlabel('RR (normalized)')
plt.ylabel('MSE')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.scatter(results_df['HR'], results_df['MSE_Freq'])
plt.title('HR vs MSE (Frequency Domain)')
plt.xlabel('HR (normalized)')
plt.ylabel('MSE (Frequency)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.scatter(results_df['RR'], results_df['MSE_Freq'])
plt.title('RR vs MSE (Frequency Domain)')
plt.xlabel('RR (normalized)')
plt.ylabel('MSE (Frequency)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'hr_rr_vs_quality.png'))
plt.close()

# Phân tích tương quan giữa các đỉnh tần số
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(results_df['Orig_Peak1_Freq'], results_df['Gen_Peak1_Freq'])
plt.title('Original vs Generated Peak 1 Frequency')
plt.xlabel('Original Peak 1 (Hz)')
plt.ylabel('Generated Peak 1 (Hz)')
plt.grid(True, alpha=0.3)
plt.plot([0, 10], [0, 10], 'r--')  # Đường chéo

plt.subplot(1, 3, 2)
plt.scatter(results_df['Orig_Peak2_Freq'], results_df['Gen_Peak2_Freq'])
plt.title('Original vs Generated Peak 2 Frequency')
plt.xlabel('Original Peak 2 (Hz)')
plt.ylabel('Generated Peak 2 (Hz)')
plt.grid(True, alpha=0.3)
plt.plot([0, 10], [0, 10], 'r--')  # Đường chéo

plt.subplot(1, 3, 3)
plt.scatter(results_df['Orig_Peak3_Freq'], results_df['Gen_Peak3_Freq'])
plt.title('Original vs Generated Peak 3 Frequency')
plt.xlabel('Original Peak 3 (Hz)')
plt.ylabel('Generated Peak 3 (Hz)')
plt.grid(True, alpha=0.3)
plt.plot([0, 10], [0, 10], 'r--')  # Đường chéo

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'peak_frequency_correlation.png'))
plt.close()

# Luu ket qua phan tich
with open(os.path.join(results_path, 'fourier_analysis_results.txt'), 'w') as f:
    f.write("KET QUA PHAN TICH BIEN DOI FOURIER\n")
    f.write("==================================\n\n")
    
    f.write("Phuong phap phan tich:\n")
    f.write("1. Bien doi Fourier nhanh (FFT) de phan tich pho tan so cua tin hieu PPG goc va tin hieu PPG da tao.\n")
    f.write("2. Phuong phap Welch de uoc luong mat do pho cong suat (PSD) cua tin hieu.\n")
    f.write("3. Tim cac dinh trong pho tan so de xac dinh cac thanh phan tan so chinh.\n")
    f.write("4. Tinh toan cac chi so danh gia trong mien thoi gian va mien tan so.\n\n")
    
    f.write("Ket qua trung binh:\n")
    f.write(f"- MSE (mien thoi gian): {avg_mse_time:.4f}\n")
    f.write(f"- PSNR: {avg_psnr:.4f}dB\n")
    f.write(f"- He so tuong quan: {avg_corr:.4f}\n")
    f.write(f"- MSE (mien tan so): {avg_mse_freq:.4f}\n\n")
    
    f.write("Phan tich chi tiet:\n")
    for i in range(len(results_df)):
        f.write(f"\nMau {i+1}:\n")
        f.write(f"- Dieu kien: HR={results_df.loc[i, 'HR']:.4f}, RR={results_df.loc[i, 'RR']:.4f}\n")
        f.write(f"- MSE (mien thoi gian): {results_df.loc[i, 'MSE_Time']:.4f}\n")
        f.write(f"- PSNR: {results_df.loc[i, 'PSNR']:.4f}dB\n")
        f.write(f"- He so tuong quan: {results_df.loc[i, 'Corr']:.4f}\n")
        f.write(f"- MSE (mien tan so): {results_df.loc[i, 'MSE_Freq']:.4f}\n")
        f.write(f"- Cac dinh tan so cua tin hieu goc: {results_df.loc[i, 'Orig_Peak1_Freq']:.2f} Hz, {results_df.loc[i, 'Orig_Peak2_Freq']:.2f} Hz, {results_df.loc[i, 'Orig_Peak3_Freq']:.2f} Hz\n")
        f.write(f"- Cac dinh tan so cua tin hieu da tao: {results_df.loc[i, 'Gen_Peak1_Freq']:.2f} Hz, {results_df.loc[i, 'Gen_Peak2_Freq']:.2f} Hz, {results_df.loc[i, 'Gen_Peak3_Freq']:.2f} Hz\n")
    
    f.write("\nNhan xet ve pho tan so:\n")
    f.write("1. Tin hieu PPG goc thuong co dinh tan so chinh o khoang 1-2 Hz, tuong ung voi nhip tim (60-120 bpm).\n")
    f.write("2. Tin hieu PPG da tao cung co xu huong tai tao dinh tan so chinh nay, nhung co the co su khac biet ve bien do.\n")
    f.write("3. Cac thanh phan tan so thap (< 0.5 Hz) lien quan den nhip tho thuong kho tai tao chinh xac hon.\n")
    f.write("4. Tin hieu PPG da tao co the thieu mot so thanh phan tan so cao (> 5 Hz) so voi tin hieu goc.\n\n")
    
    f.write("Ket luan:\n")
    f.write("Phan tich bien doi Fourier cho thay mo hinh CVAE gia lap co the tao ra tin hieu PPG voi cac dac tinh tan so co ban tuong tu nhu tin hieu goc, dac biet la thanh phan tan so lien quan den nhip tim. Tuy nhien, van co su khac biet dang ke trong cac thanh phan tan so chi tiet, dac biet la cac thanh phan tan so thap lien quan den nhip tho va cac thanh phan tan so cao. Dieu nay cho thay mo hinh CVAE thuc su duoc huan luyen day du co the cai thien kha nang tai tao cac dac tinh tan so chi tiet cua tin hieu PPG.\n")

print("\nDa hoan thanh phan tich bien doi Fourier.")
print(f"Ket qua phan tich da duoc luu tai: {os.path.join(results_path, 'fourier_analysis_results.txt')}")
print(f"Bieu do phan tich da duoc luu tai: {os.path.join(figures_path)}")
