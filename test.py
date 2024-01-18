import torch
from torch.autograd import Variable
import time
from model import AlexNet
from torchvision import transforms
import cv2
from PIL import Image

C_TIME = 100  # 摄像头持续时间
IMAGE_SIZE = [227, 227]
TEST_ROOT = 0#'./MyImage/35.jpg'


def cap_recognition(model, test_root, c_time=10):
    if test_root == 0:
        cap = cv2.VideoCapture(test_root)
        classfier = cv2.CascadeClassifier(
            "D:/My Program/python/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
        cv2.namedWindow("window", cv2.WINDOW_NORMAL)  # 创建窗口
        i_time = time.time()
        while time.time() - i_time < c_time:
            ok, frame = cap.read()  # 读取一帧数据
            grey = cv2.cvtColor(frame,
                                cv2.COLOR_BGR2GRAY)  # 将当前帧图像转换成灰度图像,利于分类器识别
            facerects = classfier.detectMultiScale(grey,
                                                   scaleFactor=1.2,
                                                   minNeighbors=3,
                                                   minSize=(32, 32))
            if len(facerects) > 0:  # 大于0则检测到人脸
                num = 1
                for faceRect in facerects:  # 单独框出每一张人脸
                    x, y, w, h = faceRect
                    img = frame[y-10:y+h+10, x-10:x+w+10]
                    # Img = Image.fromarray(img)
                    # Img.show()  # cv2 RGB通道为BGR 影响效果
                    img_x = img_to_x(img)
                    acc, faceid = test(model, img_x)

                    if acc > 0:
                        faceid = faceid
                    else:
                        faceid = "?"
                        acc = "0"
                    """
                    cv2.rectangle(frame,  # 绘制矩形人脸区域
                                  pt1=(x, y),
                                  pt2=(x+w, y+h),
                                  color=[0, 0, 255],
                                  thickness=2)
                    """
                    cv2.circle(img=frame,  # 绘制圆形人脸区域
                               center=(x + w // 2, y + h // 2),
                               radius=w // 2,
                               color=[0, 255, 0],
                               thickness=2)
                    cv2.putText(img=frame,
                                text='num:{:}, id:{:}, acc:{:6}'.format(num, faceid, acc),
                                org=(x + 30, y + 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # 显示当前捕捉到了多少人脸图片,
                                fontScale=0.5,
                                color=(255, 0, 255),
                                thickness=1)
                    num += 1
            cv2.imshow("window", frame)
            if cv2.waitKey(1) == 27:  # 27为Esc键
                break
        cap.release()
        cv2.destroyAllWindows()  # 释放摄像头并销毁所有窗口
    else:
        img = Image.open(test_root)
        img_x = img_to_x(img)
        acc, faceid = test(model, img_x)
        print(acc, faceid)


def test(model, x_test):
    model.eval()
    test_output = model(x_test)
    print(test_output)
    _, pred_y = torch.max(test_output, 1)
    acc = round(test_output[0].max().item(), 3)
    faceid = pred_y[0].item()
    return acc, faceid


def img_to_x(img):  # 对于array数组或者PIL图像使用
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(IMAGE_SIZE, antialias=True)])
    img = transform(img)
    img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    test_x = img
    return test_x


if __name__ == '__main__':
    net1 = AlexNet()
    net1.load_state_dict(torch.load('best_model_alex.pth'))
    cap_recognition(net1, test_root=TEST_ROOT, c_time=C_TIME)
