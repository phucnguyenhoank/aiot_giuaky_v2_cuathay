import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from scipy.fft import fft, fftfreq
import pandas as pd

# Đường dẫn đến file dữ liệu
data_path = r'data\bidmc_data.mat'
figures_path = r'code\figures'
os.makedirs(figures_path, exist_ok=True)

# Tải dữ liệu
print("Đang tải dữ liệu từ file .mat...")
mat_data = sio.loadmat(data_path)
data = mat_data['data'][0]  # Lấy mảng chính chứa 53 bản ghi

# Khám phá cấu trúc dữ liệu
print(f"Số lượng bản ghi: {len(data)}")

# Khám phá cấu trúc chi tiết của bản ghi đầu tiên
first_record = data[0]
print("\nKhám phá cấu trúc chi tiết của bản ghi đầu tiên:")

# Kiểm tra cấu trúc của trường ppg
ppg_field = first_record['ppg'][0, 0]
print(f"Cấu trúc của trường ppg: {type(ppg_field)}")
if hasattr(ppg_field, 'dtype') and hasattr(ppg_field.dtype, 'names'):
    print(f"Các trường con của ppg: {ppg_field.dtype.names}")

# Kiểm tra cấu trúc của trường ref
ref_field = first_record['ref'][0, 0]
print(f"Cấu trúc của trường ref: {type(ref_field)}")
if hasattr(ref_field, 'dtype') and hasattr(ref_field.dtype, 'names'):
    print(f"Các trường con của ref: {ref_field.dtype.names}")
    
    # Kiểm tra cấu trúc của trường params trong ref
    if 'params' in ref_field.dtype.names:
        params_field = ref_field['params'][0, 0]
        print(f"Cấu trúc của trường params: {type(params_field)}")
        if hasattr(params_field, 'dtype') and hasattr(params_field.dtype, 'names'):
            print(f"Các trường con của params: {params_field.dtype.names}")

# Kiểm tra chi tiết hơn về cấu trúc dữ liệu
print("\nKiểm tra chi tiết hơn về cấu trúc dữ liệu:")
print(f"Kiểu dữ liệu của ppg.v: {type(first_record['ppg'][0, 0]['v'])}")
print(f"Kích thước của ppg.v: {first_record['ppg'][0, 0]['v'].shape}")
print(f"Kiểu dữ liệu của ref.params.hr: {type(first_record['ref'][0, 0]['params'][0, 0]['hr'])}")
print(f"Kích thước của ref.params.hr: {first_record['ref'][0, 0]['params'][0, 0]['hr'].shape}")

# Trích xuất và vẽ tín hiệu PPG từ bản ghi đầu tiên
try:
    # Truy cập trực tiếp vào dữ liệu PPG
    ppg_data = first_record['ppg'][0, 0]['v']
    if isinstance(ppg_data, np.ndarray):
        # Nếu là mảng numpy, lấy dữ liệu trực tiếp
        sample_ppg = ppg_data.flatten()
    else:
        # Nếu không phải mảng numpy, chuyển đổi thành mảng
        sample_ppg = np.array(ppg_data, dtype=float).flatten()
    
    # Lấy tần số lấy mẫu
    sample_fs_ppg = float(first_record['ppg'][0, 0]['fs'][0, 0])
    
    # Tương tự cho ECG và Resp
    ecg_data = first_record['ekg'][0, 0]['v']
    if isinstance(ecg_data, np.ndarray):
        sample_ecg = ecg_data.flatten()
    else:
        sample_ecg = np.array(ecg_data, dtype=float).flatten()
    
    sample_fs_ecg = float(first_record['ekg'][0, 0]['fs'][0, 0])
    
    resp_data = first_record['ref'][0, 0]['resp_sig'][0, 0]['imp'][0, 0]['v']
    if isinstance(resp_data, np.ndarray):
        sample_resp = resp_data.flatten()
    else:
        sample_resp = np.array(resp_data, dtype=float).flatten()
    
    sample_fs_resp = float(first_record['ref'][0, 0]['resp_sig'][0, 0]['imp'][0, 0]['fs'][0, 0])

    print(f"\nTần số lấy mẫu PPG: {sample_fs_ppg} Hz")
    print(f"Tần số lấy mẫu ECG: {sample_fs_ecg} Hz")
    print(f"Tần số lấy mẫu Resp: {sample_fs_resp} Hz")

    print(f"Độ dài tín hiệu PPG: {len(sample_ppg)} mẫu")
    print(f"Độ dài tín hiệu ECG: {len(sample_ecg)} mẫu")
    print(f"Độ dài tín hiệu Resp: {len(sample_resp)} mẫu")

    # Tính thời gian cho trục x
    time_ppg = np.arange(len(sample_ppg)) / sample_fs_ppg
    time_ecg = np.arange(len(sample_ecg)) / sample_fs_ecg
    time_resp = np.arange(len(sample_resp)) / sample_fs_resp

    # Vẽ tín hiệu mẫu
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(time_ecg[:1000], sample_ecg[:1000])
    plt.title('ECG Signal (First Record - First 1000 samples)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(time_ppg[:1000], sample_ppg[:1000])
    plt.title('PPG Signal (First Record - First 1000 samples)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.plot(time_resp[:1000], sample_resp[:1000])
    plt.title('Respiratory Signal (First Record - First 1000 samples)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'sample_signals.png'))
    plt.close()
    
    # Phân tích phổ tần số của tín hiệu PPG
    def plot_fft(signal, fs, title, filename):
        N = len(signal)
        T = 1.0 / fs
        yf = fft(signal)
        xf = fftfreq(N, T)[:N//2]
        
        plt.figure(figsize=(10, 6))
        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.grid(True, alpha=0.3)
        plt.title(f'FFT of {title}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, 5)  # Giới hạn tần số hiển thị đến 5Hz
        plt.savefig(os.path.join(figures_path, filename))
        plt.close()

    # Phân tích phổ tần số của tín hiệu PPG, ECG và Resp
    plot_fft(sample_ppg, sample_fs_ppg, 'PPG Signal', 'ppg_fft.png')
    plot_fft(sample_ecg, sample_fs_ecg, 'ECG Signal', 'ecg_fft.png')
    plot_fft(sample_resp, sample_fs_resp, 'Respiratory Signal', 'resp_fft.png')
    
except Exception as e:
    print(f"Lỗi khi vẽ tín hiệu mẫu: {e}")

# Thu thập HR và RR từ tất cả các bản ghi
all_hr_values = []
all_rr_values = []
all_pr_values = []
all_spo2_values = []

for i in range(len(data)):
    try:
        record = data[i]
        params = record['ref'][0, 0]['params'][0, 0]
        
        # Lấy HR, RR, PR và SpO2 từ mỗi bản ghi
        hr_data = params['hr'][0]
        rr_data = params['rr'][0]
        pr_data = params['pr'][0]
        spo2_data = params['spo2'][0]
        
        # Chuyển đổi sang mảng numpy nếu cần
        if not isinstance(hr_data, np.ndarray):
            hr_data = np.array(hr_data, dtype=object)
        if not isinstance(rr_data, np.ndarray):
            rr_data = np.array(rr_data, dtype=object)
        if not isinstance(pr_data, np.ndarray):
            pr_data = np.array(pr_data, dtype=object)
        if not isinstance(spo2_data, np.ndarray):
            spo2_data = np.array(spo2_data, dtype=object)
        
        # Trích xuất giá trị từ mảng có cấu trúc nếu cần
        if hr_data.dtype.names is not None and 'v' in hr_data.dtype.names:
            hr_values = hr_data['v']
        else:
            hr_values = hr_data
            
        if rr_data.dtype.names is not None and 'v' in rr_data.dtype.names:
            rr_values = rr_data['v']
        else:
            rr_values = rr_data
            
        if pr_data.dtype.names is not None and 'v' in pr_data.dtype.names:
            pr_values = pr_data['v']
        else:
            pr_values = pr_data
            
        if spo2_data.dtype.names is not None and 'v' in spo2_data.dtype.names:
            spo2_values = spo2_data['v']
        else:
            spo2_values = spo2_data
        
        # Chuyển đổi sang kiểu float nếu có thể
        try:
            hr_values = np.array(hr_values, dtype=float)
            all_hr_values.append(hr_values)
        except:
            print(f"Không thể chuyển đổi HR của bản ghi {i} sang float")
            
        try:
            rr_values = np.array(rr_values, dtype=float)
            all_rr_values.append(rr_values)
        except:
            print(f"Không thể chuyển đổi RR của bản ghi {i} sang float")
            
        try:
            pr_values = np.array(pr_values, dtype=float)
            all_pr_values.append(pr_values)
        except:
            print(f"Không thể chuyển đổi PR của bản ghi {i} sang float")
            
        try:
            spo2_values = np.array(spo2_values, dtype=float)
            all_spo2_values.append(spo2_values)
        except:
            print(f"Không thể chuyển đổi SpO2 của bản ghi {i} sang float")
            
    except Exception as e:
        print(f"Lỗi khi xử lý bản ghi {i}: {e}")

# Kiểm tra xem có dữ liệu nào được thu thập không
if all_hr_values and all_rr_values and all_pr_values and all_spo2_values:
    try:
        # Ghép các mảng lại với nhau
        all_hr = np.concatenate([x for x in all_hr_values if x.size > 0])
        all_rr = np.concatenate([x for x in all_rr_values if x.size > 0])
        all_pr = np.concatenate([x for x in all_pr_values if x.size > 0])
        all_spo2 = np.concatenate([x for x in all_spo2_values if x.size > 0])

        # Loại bỏ các giá trị NaN
        valid_hr = all_hr[~np.isnan(all_hr)]
        valid_rr = all_rr[~np.isnan(all_rr)]
        valid_pr = all_pr[~np.isnan(all_pr)]
        valid_spo2 = all_spo2[~np.isnan(all_spo2)]

        # Tính toán thống kê cơ bản
        print("\nThống kê HR (Heart Rate):")
        print(f"Min: {np.min(valid_hr):.2f}, Max: {np.max(valid_hr):.2f}, Mean: {np.mean(valid_hr):.2f}, Std: {np.std(valid_hr):.2f}")

        print("\nThống kê RR (Respiratory Rate):")
        print(f"Min: {np.min(valid_rr):.2f}, Max: {np.max(valid_rr):.2f}, Mean: {np.mean(valid_rr):.2f}, Std: {np.std(valid_rr):.2f}")

        print("\nThống kê PR (Pulse Rate):")
        print(f"Min: {np.min(valid_pr):.2f}, Max: {np.max(valid_pr):.2f}, Mean: {np.mean(valid_pr):.2f}, Std: {np.std(valid_pr):.2f}")

        print("\nThống kê SpO2:")
        print(f"Min: {np.min(valid_spo2):.2f}, Max: {np.max(valid_spo2):.2f}, Mean: {np.mean(valid_spo2):.2f}, Std: {np.std(valid_spo2):.2f}")

        # Vẽ histogram cho HR và RR
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.hist(valid_hr, bins=30, alpha=0.7, color='blue')
        plt.axvline(np.mean(valid_hr), color='red', linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(valid_hr) + np.std(valid_hr), color='green', linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(valid_hr) - np.std(valid_hr), color='green', linestyle='dashed', linewidth=1)
        plt.title('Heart Rate Distribution')
        plt.xlabel('Heart Rate (bpm)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.hist(valid_rr, bins=30, alpha=0.7, color='green')
        plt.axvline(np.mean(valid_rr), color='red', linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(valid_rr) + np.std(valid_rr), color='blue', linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(valid_rr) - np.std(valid_rr), color='blue', linestyle='dashed', linewidth=1)
        plt.title('Respiratory Rate Distribution')
        plt.xlabel('Respiratory Rate (breaths/min)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.hist(valid_pr, bins=30, alpha=0.7, color='purple')
        plt.axvline(np.mean(valid_pr), color='red', linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(valid_pr) + np.std(valid_pr), color='orange', linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(valid_pr) - np.std(valid_pr), color='orange', linestyle='dashed', linewidth=1)
        plt.title('Pulse Rate Distribution')
        plt.xlabel('Pulse Rate (bpm)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.hist(valid_spo2, bins=30, alpha=0.7, color='orange')
        plt.axvline(np.mean(valid_spo2), color='red', linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(valid_spo2) + np.std(valid_spo2), color='purple', linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(valid_spo2) - np.std(valid_spo2), color='purple', linestyle='dashed', linewidth=1)
        plt.title('SpO2 Distribution')
        plt.xlabel('SpO2 (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, 'vital_signs_distribution.png'))
        plt.close()

        # Lưu thống kê vào file CSV
        stats_data = {
            'Metric': ['Heart Rate (bpm)', 'Respiratory Rate (breaths/min)', 'Pulse Rate (bpm)', 'SpO2 (%)'],
            'Min': [np.min(valid_hr), np.min(valid_rr), np.min(valid_pr), np.min(valid_spo2)],
            'Max': [np.max(valid_hr), np.max(valid_rr), np.max(valid_pr), np.max(valid_spo2)],
            'Mean': [np.mean(valid_hr), np.mean(valid_rr), np.mean(valid_pr), np.mean(valid_spo2)],
            'Std': [np.std(valid_hr), np.std(valid_rr), np.std(valid_pr), np.std(valid_spo2)],
            'Mean-Std': [np.mean(valid_hr)-np.std(valid_hr), np.mean(valid_rr)-np.std(valid_rr), 
                        np.mean(valid_pr)-np.std(valid_pr), np.mean(valid_spo2)-np.std(valid_spo2)],
            'Mean+Std': [np.mean(valid_hr)+np.std(valid_hr), np.mean(valid_rr)+np.std(valid_rr), 
                        np.mean(valid_pr)+np.std(valid_pr), np.mean(valid_spo2)+np.std(valid_spo2)]
        }

        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(os.path.join(figures_path, 'vital_signs_statistics.csv'), index=False)

        print("\nPhân tích dữ liệu hoàn tất. Các biểu đồ và thống kê đã được lưu vào thư mục figures.")

        # Tạo báo cáo tóm tắt
        with open(os.path.join(figures_path, 'data_exploration_summary.txt'), 'w') as f:
            f.write("BÁO CÁO KHÁM PHÁ DỮ LIỆU BIDMC PPG AND RESPIRATION DATASET\n")
            f.write("==========================================================\n\n")
            
            f.write(f"Số lượng bản ghi: {len(data)}\n\n")
            
            f.write("Cấu trúc dữ liệu:\n")
            f.write("- Mỗi bản ghi chứa các trường: ppg, ekg, ref, fix\n")
            f.write("- Tín hiệu PPG và ECG được lưu trữ với giá trị (v) và tần số lấy mẫu (fs)\n")
            f.write("- Tín hiệu hô hấp được lưu trữ trong trường ref.resp_sig.imp\n")
            f.write("- Các thông số sinh lý (HR, RR, PR, SpO2) được lưu trữ trong trường ref.params\n\n")
            
            f.write(f"Tần số lấy mẫu PPG: {sample_fs_ppg} Hz\n")
            f.write(f"Tần số lấy mẫu ECG: {sample_fs_ecg} Hz\n")
            f.write(f"Tần số lấy mẫu Resp: {sample_fs_resp} Hz\n\n")
            
            f.write("Thống kê HR (Heart Rate):\n")
            f.write(f"Min: {np.min(valid_hr):.2f}, Max: {np.max(valid_hr):.2f}, Mean: {np.mean(valid_hr):.2f}, Std: {np.std(valid_hr):.2f}\n")
            f.write(f"Phạm vi 1-sigma: {np.mean(valid_hr)-np.std(valid_hr):.2f} - {np.mean(valid_hr)+np.std(valid_hr):.2f}\n\n")
            
            f.write("Thống kê RR (Respiratory Rate):\n")
            f.write(f"Min: {np.min(valid_rr):.2f}, Max: {np.max(valid_rr):.2f}, Mean: {np.mean(valid_rr):.2f}, Std: {np.std(valid_rr):.2f}\n")
            f.write(f"Phạm vi 1-sigma: {np.mean(valid_rr)-np.std(valid_rr):.2f} - {np.mean(valid_rr)+np.std(valid_rr):.2f}\n\n")
            
            f.write("Thống kê PR (Pulse Rate):\n")
            f.write(f"Min: {np.min(valid_pr):.2f}, Max: {np.max(valid_pr):.2f}, Mean: {np.mean(valid_pr):.2f}, Std: {np.std(valid_pr):.2f}\n")
            f.write(f"Phạm vi 1-sigma: {np.mean(valid_pr)-np.std(valid_pr):.2f} - {np.mean(valid_pr)+np.std(valid_pr):.2f}\n\n")
            
            f.write("Thống kê SpO2:\n")
            f.write(f"Min: {np.min(valid_spo2):.2f}, Max: {np.max(valid_spo2):.2f}, Mean: {np.mean(valid_spo2):.2f}, Std: {np.std(valid_spo2):.2f}\n")
            f.write(f"Phạm vi 1-sigma: {np.mean(valid_spo2)-np.std(valid_spo2):.2f} - {np.mean(valid_spo2)+np.std(valid_spo2):.2f}\n\n")
            
            f.write("Các file đã tạo:\n")
            f.write("1. sample_signals.png - Biểu đồ mẫu của tín hiệu ECG, PPG và Respiratory\n")
            f.write("2. vital_signs_distribution.png - Phân phối của HR, RR, PR và SpO2\n")
            f.write("3. ppg_fft.png, ecg_fft.png, resp_fft.png - Phân tích phổ tần số của các tín hiệu\n")
            f.write("4. vital_signs_statistics.csv - Thống kê chi tiết của các dấu hiệu sinh tồn\n")
    except Exception as e:
        print(f"Lỗi khi xử lý thống kê: {e}")
else:
    print("Không có đủ dữ liệu để<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>")