from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
import pytesseract
import imutils
import cv2
import re
import requests
import numpy as np
import sys


def make_scan_image(image, width, ksize=(5,5), min_threshold=50, max_threshold=200):
  #image.copy()로 이미지 인자 받아오기
  org_image = image.copy()
  image = imutils.resize(image, width=width)
  ratio = org_image.shape[1] / float(image.shape[1])
 
  # 이미지를 grayscale로 변환하고 blur를 적용
  # 모서리를 찾기위한 이미지 연산
  # 이미지를 grayscale로 변환하고, 흑백 2색으로 변환
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, ksize, 0)
  ret, binary = cv2.threshold(blurred,155,255,cv2.THRESH_BINARY)
  binary = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)
  
  # contours를 찾아 크기순으로 정렬
  cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

  findCnt = None
  

  # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
  # arcLength는 외곽선 길이를 구하는 함수
  #cnts는 외곽선들이 모여있는 점들의 집합의 집합
  for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
 
 
    # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
    if len(approx) == 4:
      findCnt = approx
      break
 
  # 만약 추출한 윤곽이 없을 경우 오류
  if findCnt is None:
    print("윤곽을 찾을 수 없습니다")
    raise Exception(("Could not find outline."))

  
  #윤곽선 그리기
  output = image.copy()
  cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)
  
  # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
  receipt = four_point_transform(org_image, findCnt.reshape(4, 2) * ratio)

  #원하는 영역 추출로 변환과정
  #grayscale로 변환
  gray = cv2.cvtColor(receipt, cv2.COLOR_BGR2GRAY)
  (H, W) = gray.shape

  # getStructuringElement -> 형태학적 변환 인자로 커널사이즈 조정
  #이미지 필터링과정
  rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,10))
  sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
  gray = cv2.GaussianBlur(gray, ksize, 0)
  blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
  grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
  grad = np.absolute(grad)
  (minVal, maxVal) = (np.min(grad), np.max(grad))
  grad = (grad - minVal) / (maxVal - minVal)
  grad = (grad * 255).astype("uint8")
 #close
  grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
  thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 #close + Dilation(팽창) + Erosion(침식)
  close_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
  close_thresh = cv2.dilate(close_thresh,None, iterations=6)
  close_thresh = cv2.erode(close_thresh, None, iterations=2)

  #Detection
  #contour 찾기
  cnts = cv2.findContours(close_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  cnts = imutils.grab_contours(cnts)
  cnts = sort_contours(cnts, method="top-to-bottom")[0]
 
  roi_list = []
  margin = 50
 
  for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    #cnts중 특정영역에 해당하는 부분만 따로 표시
    if (w*2) > x and (2*h/5) < y:
      color = (0, 255, 0)
      roi = receipt[y-margin:y + h + margin, x:x + w + margin]
      #크기가 0인경우 에러를 야기하므로 if문 처리로 회피
      if roi.size == 0:
        continue
      roi_list.append(roi)
    else:
      color = (0, 0, 255)
    
  print("[INFO] OCR결과:")
  
  #텍스트 출력
  for roi in roi_list: 
    #roi_list는 좌표값을 의미한다
    gray_roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    threshold_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    roi_text = pytesseract.image_to_string(roi, lang='kor+eng')
    if isHangul(roi_text):
      showProduct(roi_text)

"""      
def isProduct(text):
  if text.find("00") == 0:
    return 1
  elif "상품" in text:
    return 1
"""

def showProduct(text):
  print(text)

#한글을 check하는 부분
def isHangul(text):
    #Check the Python Version
    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', text))
    return hanCount > 0


url = 'http://youbojob.why-be.co.kr/data/file/paper/thumb-2944312436_0YM4PlBE_0f464cf018a159aa3ad9b03a31f3c6d766d04905_600x800.jpg'
 
image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
org_image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR) 
make_scan_image(org_image, width=200, ksize=(5, 5))

