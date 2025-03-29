import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

# Đường dẫn đến file dữ liệu
data_path = r'data/bidmc_data.mat'
processed_data_path = r'data/processed'
figures_path = r'code/figures'

# Tạo thư mục nếu chưa tồn tại
os.makedirs(processed_data_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# Tải dữ liệu
print("Đang tải dữ liệu từ file .mat...")
mat_data = sio.loadmat(data_path)
data = mat_data['data'][0]  # Lấy mảng chính chứa 53 bản ghi

print(f"Số lượng bản ghi: {len(data)}")

# Hàm lọc nhiễu cho tín hiệu PPG
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Hàm chuẩn hóa tín hiệu
def normalize_signal(signal, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
        signal_reshaped = signal.reshape(-1, 1)
        normalized = scaler.fit_transform(signal_reshaped).flatten()
    elif method == 'standard':
        scaler = StandardScaler()
        signal_reshaped = signal.reshape(-1, 1)
        normalized = scaler.fit_transform(signal_reshaped).flatten()
    elif method == 'simple':
        normalized = (signal - np.mean(signal)) / np.std(signal)
    else:
        raise ValueError("Phương pháp chuẩn hóa không hợp lệ")
    return normalized

# Hàm chia tín hiệu thành các đoạn có độ dài cố định
def segment_signal(signal, segment_length, overlap=0):
    step = int(segment_length * (1 - overlap))
    segments = []
    for i in range(0, len(signal) - segment_length + 1, step):
        segments.append(signal[i:i + segment_length])
    return np.array(segments)

# Hàm trích xuất đặc trưng HR và BR từ tín hiệu
def extract_hr_br_features(hr_values, rr_values):
    # Lấy giá trị trung bình của HR và BR
    hr_mean = np.mean(hr_values)
    rr_mean = np.mean(rr_values)
    
    # Chuẩn hóa HR và BR về khoảng [0, 1]
    hr_normalized = hr_mean / 200.0  # Giả sử HR tối đa là 200 bpm
    rr_normalized = rr_mean / 60.0   # Giả sử RR tối đa là 60 breaths/min
    
    return hr_normalized, rr_normalized

# Danh sách để lưu trữ dữ liệu đã tiền xử lý
ppg_segments = []
hr_features = []
rr_features = []

# Tham số tiền xử lý
fs = 125  # Tần số lấy mẫu (Hz)
segment_length = 8 * fs  # Độ dài đoạn tín hiệu (8 giây)
overlap = 0.5  # Độ chồng lấp giữa các đoạn (50%)
lowcut = 0.5  # Tần số cắt dưới cho bộ lọc (Hz)
highcut = 8.0  # Tần số cắt trên cho bộ lọc (Hz)

# Tiền xử lý dữ liệu từ mỗi bản ghi
valid_records = 0
for i in range(len(data)):
    try:
        record = data[i]
        
        # Trích xuất tín hiệu PPG
        ppg_data = record['ppg'][0, 0]['v']
        if isinstance(ppg_data, np.ndarray):
            ppg_signal = ppg_data.flatten()
        else:
            ppg_signal = np.array(ppg_data, dtype=float).flatten()
        
        # Trích xuất HR và RR - Truy cập trực tiếp vào giá trị số trong mảng
        try:
            # Trích xuất HR từ mảng lồng nhau
            hr_data = record['ref'][0, 0]['params'][0, 0]['hr'][0]
            if hasattr(hr_data, 'dtype') and hr_data.dtype.names is not None and 'v' in hr_data.dtype.names:
                hr_values_raw = hr_data['v']
            else:
                hr_values_raw = hr_data
                
            # Trích xuất RR từ mảng lồng nhau
            rr_data = record['ref'][0, 0]['params'][0, 0]['rr'][0]
            if hasattr(rr_data, 'dtype') and rr_data.dtype.names is not None and 'v' in rr_data.dtype.names:
                rr_values_raw = rr_data['v']
            else:
                rr_values_raw = rr_data
            
            # Trích xuất giá trị số từ mảng lồng nhau
            hr_values = []
            for item in hr_values_raw:
                if isinstance(item, np.ndarray) and item.size > 0:
                    hr_values.append(float(item[0]))
                elif np.isscalar(item):
                    hr_values.append(float(item))
            
            rr_values = []
            for item in rr_values_raw:
                if isinstance(item, np.ndarray) and item.size > 0:
                    rr_values.append(float(item[0]))
                elif np.isscalar(item):
                    rr_values.append(float(item))
            
            # Chuyển đổi sang mảng numpy
            hr_values = np.array(hr_values)
            rr_values = np.array(rr_values)
            
            # Kiểm tra xem có đủ dữ liệu không
            if len(hr_values) == 0 or len(rr_values) == 0:
                print(f"Không đủ dữ liệu HR/RR cho bản ghi {i}, bỏ qua bản ghi này")
                continue
                
        except Exception as e:
            print(f"Lỗi khi trích xuất HR/RR của bản ghi {i}: {e}, bỏ qua bản ghi này")
            continue
        
        # Lọc nhiễu tín hiệu PPG
        ppg_filtered = butter_bandpass_filter(ppg_signal, lowcut, highcut, fs)
        
        # Chuẩn hóa tín hiệu PPG
        ppg_normalized = normalize_signal(ppg_filtered, method='minmax')
        
        # Chia tín hiệu thành các đoạn
        segments = segment_signal(ppg_normalized, segment_length, overlap)
        
        # Trích xuất đặc trưng HR và BR cho mỗi đoạn
        for segment in segments:
            hr_feature, rr_feature = extract_hr_br_features(hr_values, rr_values)
            
            # Thêm vào danh sách
            ppg_segments.append(segment)
            hr_features.append(hr_feature)
            rr_features.append(rr_feature)
        
        valid_records += 1
        print(f"Đã xử lý bản ghi {i}, số đoạn tín hiệu: {len(segments)}")
        
    except Exception as e:
        print(f"Lỗi khi xử lý bản ghi {i}: {e}")

print(f"\nĐã xử lý thành công {valid_records}/{len(data)} bản ghi")
print(f"Tổng số đoạn tín hiệu: {len(ppg_segments)}")

# Kiểm tra xem có đủ dữ liệu không
if len(ppg_segments) == 0:
    print("Không có đủ dữ liệu để tiếp tục. Sử dụng dữ liệu giả lập để minh họa.")
    
    # Tạo dữ liệu giả lập để minh họa
    num_samples = 100
    ppg_segments = np.random.randn(num_samples, segment_length) * 0.1
    hr_features = np.random.uniform(0.3, 0.6, num_samples)  # HR từ 60-120 bpm (chuẩn hóa)
    rr_features = np.random.uniform(0.1, 0.3, num_samples)  # RR từ 6-18 breaths/min (chuẩn hóa)
    
    print(f"Đã tạo {num_samples} mẫu dữ liệu giả lập.")

# Chuyển đổi danh sách thành mảng numpy
ppg_segments = np.array(ppg_segments)
hr_features = np.array(hr_features)
rr_features = np.array(rr_features)

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, hr_train, hr_test, rr_train, rr_test = train_test_split(
    ppg_segments, hr_features, rr_features, test_size=0.2, random_state=42
)

# Lưu dữ liệu đã tiền xử lý
np.save(os.path.join(processed_data_path, 'ppg_train.npy'), X_train)
np.save(os.path.join(processed_data_path, 'ppg_test.npy'), X_test)
np.save(os.path.join(processed_data_path, 'hr_train.npy'), hr_train)
np.save(os.path.join(processed_data_path, 'hr_test.npy'), hr_test)
np.save(os.path.join(processed_data_path, 'rr_train.npy'), rr_train)
np.save(os.path.join(processed_data_path, 'rr_test.npy'), rr_test)

# Lưu thông tin về dữ liệu đã tiền xử lý
with open(os.path.join(processed_data_path, 'preprocessing_info.txt'), 'w') as f:
    f.write("THONG TIN TIEN XU LY DU LIEU\n")
    f.write("============================\n\n")
    
    f.write(f"So luong ban ghi da xu ly: {valid_records}/{len(data)}\n")
    f.write(f"Tong so doan tin hieu: {len(ppg_segments)}\n\n")
    
    f.write("Tham so tien xu ly:\n")
    f.write(f"- Tan so lay mau: {fs} Hz\n")
    f.write(f"- Do dai doan tin hieu: {segment_length} mau ({segment_length/fs} giay)\n")
    f.write(f"- Do chong lap: {overlap*100}%\n")
    f.write(f"- Tan so cat duoi: {lowcut} Hz\n")
    f.write(f"- Tan so cat tren: {highcut} Hz\n\n")
    
    f.write("Kich thuoc tap du lieu:\n")
    f.write(f"- Tap huan luyen: {X_train.shape[0]} mau\n")
    f.write(f"- Tap kiem thu: {X_test.shape[0]} mau\n\n")
    
    f.write("Thong ke HR (chuan hoa):\n")
    f.write(f"- Min: {np.min(hr_features):.4f}, Max: {np.max(hr_features):.4f}\n")
    f.write(f"- Mean: {np.mean(hr_features):.4f}, Std: {np.std(hr_features):.4f}\n\n")
    
    f.write("Thong ke RR (chuan hoa):\n")
    f.write(f"- Min: {np.min(rr_features):.4f}, Max: {np.max(rr_features):.4f}\n")
    f.write(f"- Mean: {np.mean(rr_features):.4f}, Std: {np.std(rr_features):.4f}\n")
    
    if valid_records == 0:
        f.write("\nLuu y: Du lieu duoc su dung la du lieu gia lap do khong the trich xuat duoc du lieu thuc tu bo du lieu BIDMC.\n")


# Vẽ biểu đồ phân phối HR và RR đã chuẩn hóa
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(hr_features, bins=30, alpha=0.7, color='blue')
plt.axvline(np.mean(hr_features), color='red', linestyle='dashed', linewidth=1)
plt.title('Normalized Heart Rate Distribution')
plt.xlabel('Normalized Heart Rate')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(rr_features, bins=30, alpha=0.7, color='green')
plt.axvline(np.mean(rr_features), color='red', linestyle='dashed', linewidth=1)
plt.title('Normalized Respiratory Rate Distribution')
plt.xlabel('Normalized Respiratory Rate')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'normalized_hr_rr_distribution.png'))
plt.close()

# Vẽ một số đoạn tín hiệu PPG đã tiền xử lý
plt.figure(figsize=(15, 10))

for i in range(min(5, len(X_train))):
    plt.subplot(5, 1, i+1)
    plt.plot(X_train[i])
    plt.title(f'Preprocessed PPG Segment {i+1}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'preprocessed_ppg_segments.png'))
plt.close()

print("\nTien xy ly du lieu hoan tat. Du lieu da duoc luu vao thu muc processed.")
