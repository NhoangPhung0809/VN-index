import os
import sys
import torch

print("--- BẮT ĐẦU KIỂM TRA CHRONOS ---")

# 1. Kiểm tra PyTorch
print(f"[1/3] PyTorch version: {torch.__version__} (CUDA: {torch.cuda.is_available()})")

# 2. Kiểm tra Import Chronos
try:
    from chronos import ChronosPipeline
    print("[2/3] Import Chronos thành công!")
except ImportError as e:
    print(f"LỖI IMPORT: {e}")
    sys.exit()

# 3. Thử tải Model (Bước quan trọng nhất)
print("[3/3] Đang tải Model 'amazon/chronos-t5-tiny' (Chờ khoảng 30s)...")
try:
    try:
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cpu",  # Ép chạy CPU để test cho chắc
            torch_dtype=torch.float32,
        )
        print(">>> THÀNH CÔNG (Online)! Model đã tải xong.")
    except Exception as e_online:
        print(f">>> Online thất bại ({e_online}), chuyển sang chế độ Offline...")
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
            local_files_only=True
        )
        print(">>> THÀNH CÔNG (Offline)! Model đã tải từ cache.")
        
    print(">>> Bạn có thể bật App dự báo được rồi.")
except Exception as e:
    print(f">>> THẤT BẠI CẢ 2 CHẾ ĐỘ: {e}")
    print("Gợi ý: Kiểm tra kết nối mạng (để tải từ HuggingFace) hoặc RAM máy tính.")