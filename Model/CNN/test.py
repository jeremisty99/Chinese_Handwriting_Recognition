import base64
import torch
from torchvision import transforms
from train import Net
from ..projection_detect import locate_process_projection_single, locate_process_projection_multiple
from cv2 import cv2

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, ], [0.229, ])
])
pth_path = r'model.pth'
if torch.cuda.is_available():
    net = Net().eval().cuda()
else:
    net = Net().eval()  # 评估
net.load_state_dict(torch.load(pth_path))
result_list = ["一", "业", "北", "大", "子", "学", "家", "工", "恒", "杨", "松", "河", "涵", "王", "赵", "路"]


def predict_single(image):
    """
    :param image_path: 图片路径
    """
    img = locate_process_projection_single(image)
    # plt.imshow(img)
    # plt.show()
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        if torch.cuda.is_available():
            images = img.cuda()
        else:
            images = img
        outputs = net(images)
        # 取得分最高的那个类
        value, predicted = torch.max(outputs.data, 1)
        predicted_list = outputs.data.cpu().numpy()[0][0:16].tolist()
        predicted_list = [float('{:.2f}'.format(i)) for i in predicted_list]
        result_index = predicted.cpu().item()
        result_dict = {}
        for index, c in enumerate(result_list):
            result_dict[c] = predicted_list[index]
        result_dict = sorted(result_dict.items(), key=lambda item: item[1], reverse=True)
        return result_dict, result_list[result_index]


# def predict_multiple(image):
#     img_list, rec_image = locate_process_projection_multiple(image)
#     result_str = ""
#     result_str_list = []
#     line_now = 0
#     for i in img_list:
#         img = transform(i[0]).unsqueeze(0)
#         line_index = i[1]
#         with torch.no_grad():
#             if torch.cuda.is_available():
#                 images = img.cuda()
#             else:
#                 images = img
#             outputs = net(images)
#             # 取得分最高的那个类
#             value, predicted = torch.max(outputs.data, 1)
#             result_index = predicted.cpu().item()
#             if line_index != line_now:
#                 result_str_list.append(result_str)
#                 result_str = result_list[result_index]
#                 line_now = line_index
#             else:
#                 result_str = result_str + result_list[result_index]
#     result_str_list.append(result_str)
#     b64_str = str(base64.b64encode(cv2.imencode('.jpg', rec_image)[1].tobytes()), encoding='utf-8')
#     return result_str_list, b64_str


if __name__ == '__main__':
    print(predict_single(cv2.imread("2.jpg")))
    print(predict_single(cv2.imread("3.jpg")))
    print(predict_single(cv2.imread("4.jpg")))
