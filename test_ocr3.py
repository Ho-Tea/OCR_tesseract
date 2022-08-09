from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
import pytesseract
import imutils
import cv2
import re
import requests
import numpy as np

#이미지를 확인하기 위한 function
def plt_imshow(title='image', img=None, figsize=(8, 5)):
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()


def make_scan_image(image, width, ksize=(5,5), min_threshold=400, max_threshold=400):
  image_list_title = []
  image_list = []

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
  # 많은 시행착오를 겪고 edged는 사용안하기로
  #edged = cv2.Canny(binary, min_threshold, max_threshold)
  
  # 순서
  image_list_title = [ 'gray','blurred','binary']
  image_list = [gray, blurred, binary]

  # contours를 찾아 크기순으로 정렬
  cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

  findCnt = None
  
  # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
  # arcLength는 외곽선 길이를 구하는 함수
  
  
  #max = 0;
  #cnts는 외곽선들이 모여있는 점들의 집합의 집합
  for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    #area = cv2.contourArea(approx)
 
    # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
    if len(approx) == 4:
      findCnt = approx
      break

    # 시행착오
    # 외곽선의 크기중 제일 큰영역을 판단하고 break
    #if area > max:
      #max = area
      #findCnt = approx
      
 
  # 만약 추출한 윤곽이 없을 경우 오류
  if findCnt is None:
    print("윤곽을 찾을 수 없습니다")
    #raise Exception(("Could not find outline."))

  
  #윤곽선 그리기
  output = image.copy()
  cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)
  image_list_title.append("Outline")
  image_list.append(output)
  plt_imshow(image_list_title, image_list)
  
  # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
  receipt = four_point_transform(org_image, findCnt.reshape(4, 2) * ratio)
 
  plt_imshow(image_list_title, image_list)
  plt_imshow("Receipt Transform", receipt)

  return org_image
  
  """
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

  cnts = cv2.findContours(close_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sort_contours(cnts, method="top-to-bottom")[0]
 
  roi_list = []
  roi_title_list = []
 
  margin = 20
  receipt_grouping = receipt.copy()
 
  for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w // float(h)
 
    if ar > 3.0 and ar < 6.5 and (W/2) < x:
      color = (0, 255, 0)
      roi = receipt[y - margin:y + h + margin, x - margin:x + w + margin]
      roi_list.append(roi)
      roi_title_list.append("Roi_{}".format(len(roi_list)))
    else:
      color = (0, 0, 255)
 
    cv2.rectangle(receipt_grouping, (x - margin, y - margin), (x + w + margin, y + h + margin), color, 2)
    cv2.putText(receipt_grouping, "".join(str(ar)), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
  
    #plt_imshow(["Grouping Image"], [receipt_grouping], figsize=(16, 10))
 
  for idx, roi in enumerate(roi_list):
    if idx == 0:
      mergeImg = mergeResize(roi)
    else:
      cropImg = mergeResize(roi)
      mergeImg = np.concatenate((mergeImg, cropImg), axis=0)

  options = "--psm 4"

  threshold_mergeImg = cv2.threshold(mergeImg, 150, 255, cv2.THRESH_BINARY)[1]
  plt_imshow(["Merge Image"], [threshold_mergeImg])

  return threshold_mergeImg


def mergeResize(img, row=300, col=200):
    IMG_COL = col #66
 
    # row값에 따른 col값 변경
    IMG_COL = int((row * IMG_COL)/row)
 
    IMG_ROW = row
    border_v = 0
    border_h = 0
 
    if (IMG_COL / IMG_ROW) >= (img.shape[0] / img.shape[1]):
        border_v = int((((IMG_COL / IMG_ROW) * img.shape[1]) - img.shape[0]) / 2)
    else:
        border_h = int((((IMG_ROW / IMG_COL) * img.shape[0]) - img.shape[1]) / 2)
    img = cv2.copyMakeBorder(img, top=border_v, bottom=border_v, left=0, right=border_h + border_h, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    img = cv2.resize(img, (IMG_ROW, IMG_COL))
    return img
  """

url = 'https://blog.kakaocdn.net/dn/q8bkq/btqJCt61orM/J4OuH0kgcyyVkPL8e76WD0/img.jpg'
 
image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
org_image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR) 
 
plt_imshow("orignal image", org_image)
receipt_image = make_scan_image(org_image, width=200, ksize=(5, 5))

options = "--psm 4"
text = pytesseract.image_to_string(cv2.cvtColor(receipt_image, cv2.COLOR_BGR2RGB), config=options)
 
# OCR결과 출력
print("[INFO] OCR결과:")
print("==================")
print(text)
print("\n")