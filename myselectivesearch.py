# 이름 MySelectiveSearch로 바꾸고 __init__추가하세요.
import data_prepare
import cv2

train_dir = data_prepare.train_dir
test_dir = data_prepare.test_dir
valid_dir = data_prepare.valid_dir
'''
print(train_dir + '\\image\\000001.jpg')
img = cv2.imread(train_dir + '\\image\\000001.jpg')
#cv2.imshow("img", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

ss.setBaseImage(img)
ss.switchToSelectiveSearchFast()
ssresult = ss.process()

gtvalue = []

for e, result in enumerate(ssresult):
    x, y, w, h = result
    gtvalue.append({"x1": x, "x2": x+w, "y1": y, "y2": y+h})

print(gtvalue[0])
'''


# img 넣어주면 자동으로 SelectiveSearch한 좌표들을 반환해줌.
def myselectivesearch(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    ssresult = ss.process()
    gtvalue = []
    for e, result in enumerate(ssresult):
        x, y, w, h = result
        gtvalue.append({"x1": x, "x2": x + w, "y1": y, "y2": y + h})
    return gtvalue
