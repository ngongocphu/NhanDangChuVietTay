"""
Batch OCR - Chỉ nhận dạng ảnh trong thư mục AnhTT
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from PIL import Image
from ocr_without_training import OCRWithoutTraining

def get_image_files(directory):
    """Lấy tất cả file ảnh trong thư mục"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    for file in os.listdir(directory):
        if Path(file).suffix in image_extensions:
            image_files.append(os.path.join(directory, file))
    
    return sorted(image_files)

def read_ground_truth(image_path):
    """Đọc ground truth từ file .txt tương ứng"""
    txt_path = image_path.rsplit('.', 1)[0] + '.txt'
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except:
            pass
    return None

def calculate_cer(predicted, ground_truth):
    """Tính Character Error Rate"""
    if not ground_truth:
        return None
    
    # Normalize
    pred = predicted.lower().strip()
    gt = ground_truth.lower().strip()
    
    # Simple CER calculation
    if pred == gt:
        return 0.0
    
    # Count differences
    min_len = min(len(pred), len(gt))
    max_len = max(len(pred), len(gt))
    
    if max_len == 0:
        return 0.0
    
    # Simple character-level comparison
    errors = 0
    for i in range(min_len):
        if pred[i] != gt[i]:
            errors += 1
    errors += abs(len(pred) - len(gt))
    
    return errors / max_len * 100

def batch_ocr_anhTT(output_file='batch_ocr_anhTT_results.json'):
    """Nhận dạng toàn bộ ảnh trong thư mục AnhTT"""
    print("=" * 70)
    print("BATCH OCR - NHẬN DẠNG THƯ MỤC AnhTT")
    print("=" * 70)
    
    directory = 'AnhTT'
    if not os.path.exists(directory):
        print(f"❌ Thư mục không tồn tại: {directory}")
        return
    
    # Khởi tạo OCR engine
    print("\n🔄 Đang khởi tạo EasyOCR...")
    try:
        ocr = OCRWithoutTraining('easyocr')
        print("✅ EasyOCR đã sẵn sàng\n")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo OCR: {e}")
        return
    
    # Lấy tất cả file ảnh
    all_images = get_image_files(directory)
    total_images = len(all_images)
    
    print(f"📁 {directory}: {total_images} ảnh")
    
    if total_images == 0:
        print("❌ Không tìm thấy ảnh nào!")
        return
    
    # Nhận dạng từng ảnh
    results = []
    correct_count = 0
    total_time = 0
    
    print("\n" + "=" * 70)
    print("BẮT ĐẦU NHẬN DẠNG...")
    print("=" * 70 + "\n")
    
    for idx, image_path in enumerate(all_images, 1):
        filename = os.path.basename(image_path)
        print(f"[{idx}/{total_images}] Đang xử lý: {filename}")
        
        # Đọc ground truth nếu có
        ground_truth = read_ground_truth(image_path)
        
        # Nhận dạng
        start_time = time.time()
        try:
            # Sử dụng PIL Image để đảm bảo tương thích
            img = Image.open(image_path)
            recognized_text = ocr.recognize(img)
            processing_time = time.time() - start_time
            total_time += processing_time
            
            # Tính CER nếu có ground truth
            cer = None
            match = False
            if ground_truth:
                cer = calculate_cer(recognized_text, ground_truth)
                match = (recognized_text.strip().lower() == ground_truth.strip().lower())
                if match:
                    correct_count += 1
            
            # Lấy confidence
            confidence = ocr.get_confidence()
            
            # Kết quả
            result = {
                'image': filename,
                'recognized_text': recognized_text,
                'ground_truth': ground_truth,
                'cer': round(cer, 2) if cer is not None else None,
                'match': match,
                'confidence': round(confidence, 1),
                'processing_time': round(processing_time, 2)
            }
            results.append(result)
            
            # Hiển thị kết quả
            status = "✅" if match else "❌"
            print(f"   {status} Kết quả: {recognized_text}")
            if ground_truth:
                print(f"   Ground truth: {ground_truth}")
                print(f"   CER: {cer:.2f}%" if cer is not None else "   CER: N/A")
            print(f"   Confidence: {confidence:.1f}% | Time: {processing_time:.2f}s\n")
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"   ❌ Lỗi: {e}\n")
            results.append({
                'image': filename,
                'error': str(e),
                'processing_time': round(processing_time, 2)
            })
    
    # Thống kê
    print("=" * 70)
    print("THỐNG KÊ KẾT QUẢ")
    print("=" * 70)
    print(f"\n📊 Tổng số ảnh: {total_images}")
    print(f"✅ Nhận dạng thành công: {len([r for r in results if 'error' not in r])}")
    
    results_with_gt = [r for r in results if r.get('ground_truth')]
    if results_with_gt:
        print(f"📝 Có ground truth: {len(results_with_gt)}")
        print(f"🎯 Accuracy: {correct_count}/{len(results_with_gt)} ({correct_count/len(results_with_gt)*100:.2f}%)")
        
        # Tính CER trung bình
        cers = [r['cer'] for r in results_with_gt if r.get('cer') is not None]
        if cers:
            avg_cer = sum(cers) / len(cers)
            print(f"📈 CER trung bình: {avg_cer:.2f}%")
    
    print(f"⏱️  Thời gian trung bình: {total_time/total_images:.2f}s/ảnh")
    print(f"⏱️  Tổng thời gian: {total_time:.2f}s")
    
    # Lưu kết quả
    output_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_images': total_images,
        'results': results,
        'statistics': {
            'successful': len([r for r in results if 'error' not in r]),
            'with_ground_truth': len(results_with_gt),
            'correct': correct_count,
            'accuracy': round(correct_count/len(results_with_gt)*100, 2) if results_with_gt else 0,
            'avg_cer': round(sum(cers)/len(cers), 2) if cers else None,
            'avg_time': round(total_time/total_images, 2)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Đã lưu kết quả vào: {output_file}")
    print("=" * 70)

if __name__ == '__main__':
    import sys
    batch_ocr_anhTT()


