# OCR_tesseract
- Module using OCR technology

## Reason:
- Naver(clova), google(drive) 의 OCR API를 이용하여 공부 할까 했지만
<br> **요금부가 및 지속적인 서비스에 애매**
- clova 대략적인 절차
<img src="1.jpeg">
- 따라서, **pytesseract**이용

## What is Tesseract?
- 다양한 운영체제를 위한 광학 문자 인식(OCR) 엔진
  - 쉽게 말해, OCR 오픈소스 라이브러
  - 3대 OCR엔진에 속한다
- 많은 언어 및 스크립트에 대한 LSTM(딥러닝)기반 OCR 엔진 및 모델이 추가되어 총 116개의 언어가 제공된다


## How to use?
- Tesseract Engine 설치
  - python에서 직접 경로 추가
- OpenCV, PyTesseract 설치
  - OCR,이미지 처리를 수행하는데 필요한 필수 python패키지 설치
- Trained Data Download
  - 한글을 인식하는 셈플데이터 `/opt/homebrew/Cellar/tesseract/5.2.0/share/tessdata` 에 저장


### Test1(test_ocr.py, test_ocr2.py)
- 입력된 이미지 전처리과정을 통해 Scan한 이미지로 변경 `plt_imshow`메서드를 통해 이미지 확인
- `Contour`과정을 통해 이미지의 외곽선을 검출<br>(`run_tesseract_ocr`메서드),(`make_scan_image`메서드)
- gray, blurred, edged의 방식으로 Scan이미지로 변경
  - (test_ocr.py, test_ocr2.py) 동일한 내용
- **이미지 연산을 통해 Scan이미지로의 변경보다는 다른 방법 강구**

<img src="2.png">

<img src="3.png">


<br/>

### Test2()
- 원하는 영역만 추출이 목적
- 이미지 처리기술과 OpenCV 라이브러리를 사용하여 입력 이미지에서 원하는 텍스트 추출
  - 이미지 연산을 통한 영역 추출
    - 그레이 스케일로 변환
    - 흐릿한 Grayscale 이미지에 blackhat 모노폴리 연산을 적용<br> blackhat연산은 밝은 배경에서 어두운 영역을 드러내기위해 사용 -> 영수증에 효율적
    - 닫힘연산을 통해 끊어져보이는 객체를 연결하여 gruoping

  ``` python
  gray = cv2.cvtColor(receipt, cv2.COLOR_BGR2GRAY)
  (H, W) = gray.shape
 
  rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 20))
  sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 21))
 
  gray = cv2.GaussianBlur(gray, (11, 11), 0)
  blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
 
  grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
  grad = np.absolute(grad)
  (minVal, maxVal) = (np.min(grad), np.max(grad))
  grad = (grad - minVal) / (maxVal - minVal)
  grad = (grad * 255).astype("uint8")
 
  grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
  thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
  close_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
  close_thresh = cv2.erode(close_thresh, None, iterations=2)
 
  plt_imshow(["Original", "Blackhat", "Gradient", "Rect Close", "Square Close"], [receipt, blackhat, grad, thresh, close_thresh], figsize=(16, 10))
 
  ```

  - Detecion
    - grouping된 영역의 윤곽선을 찾고 그 윤곽선이 특정 조건(ex: 종횡비)에 만족하는 영역만 추출
 
    

    
[reference](https://yunwoong.tistory.com/72?category=902345) 


