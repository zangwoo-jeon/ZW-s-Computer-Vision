import cv2 as cv
import numpy as np # openCV와 numpy를 import

bins = np.arange(256).reshape(256, 1)
# 0~255까지의 array를 생성한 뒤, 256개의 행 1열인 행렬로 만든다.

def draw_histogram(img):
    h = np.zeros((img.shape[0], 256), dtype=np.uint8)
    #히스토그램 칸의 크기 설정, 데이터 타입은 8바이트 자연수 값.
    hist_item = cv.calcHist([img], [0], None, [256], [0,256])
    #히스토그램 계산식 cv.calcHist(images, channels(그레이 스케일이여서 0),
    # mask(전체 이미지여서 None), histSize(막대 개수 전체여서 256), range(각 빈의 경계값)
    cv.normalize(hist_item, hist_item, 0, 255, cv.NORM_MINMAX)
    #cv.normalize(src(입력 이미지), dst(출력 이미지), alpha(최소값), beta(최대값), norm_type(정규화 진행 방식))
    # 공식 : N = (filtered - np.min()) / (np.max() - np.min())
    hist=np.int32(np.around(hist_item))
    #반올림
    for x,y in enumerate(hist):
        cv.line(h, (x,0+10), (x,y+10), (255,255,255))
        #cv.line(img(선분이 그려질 이미지), (x1,y1)선분의 시작점, (x2, y2)선분의 끝점, color선분의 색)
    cv.line(h, (0, 0+10), (0, 5), (255, 255, 255))
    cv.line(h, (255, 0+10), (255, 5), (255, 255, 255))
    y = np.flipud(h)
    #y = h행렬의 중앙의 가로축을 기준으로 뒤집음.

    hist, bin = np.histogram(img.flatten(), 256, [0, 256])
    #이미지를 1차원 배열로 변환
    cdf = hist.cumsum()
    # 누적 히스토그램
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    #정규화 (누적 명암값 * 255 / 전체 명암값)
    cv.normalize(cdf_normalized, cdf_normalized, 0, 255, cv.NORM_MINMAX)
    #normalize()는 새로운 범위로 변환해주는 역할을 한다.
    #normalize(입력값, 출력값, 변환 최소값, 변환 최대값, 변환 방식)
    hist = np.int32(np.around(cdf_normalized))
    #반올림
    pts = np.int32(np.column_stack((bins, hist)))
    pts += [257, 10]

    cv.line(h, (0+257, 0+10), (0+257, 5), (255, 255, 255))
    cv.line(h, (255+257, 0+10), (255+257, 5), (255, 255, 255))
    cv.polylines(h, [pts], False, (255, 255, 255))

    return y

img = cv.imread('bmw.PNG', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

line = draw_histogram(gray)
result1 = np.hstack((gray, line))
cv.imshow('result1', result1)

hist, bin = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_mask = np.ma.masked_equal(cdf, 0)
cdf_mask = (cdf_mask - cdf_mask.min())*255 / (cdf_mask.max()-cdf_mask.min())
#누적합의 최대값,, 최소값을 이용하여 히스토그램이 넓게 분포되도록 만듬
cdf = np.ma.filled(cdf_mask, 0).astype('uint8')
equ = cdf[gray]
# 히스토그램 평활화 이미지

line = draw_histogram(equ)
result2 = np.hstack((equ, line))
cv.imshow('result2', result2)

cv.waitKey(0)
cv.destroyAllWindows()