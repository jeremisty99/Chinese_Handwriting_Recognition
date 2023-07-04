/* global $ */
class Main {
    constructor() {
        this.canvas = document.getElementById('main');
        // this.input = document.getElementById('input');
        if (document.documentElement.clientWidth > 768) {
            // this.width = 961; // 16 * 28 + 1
            // this.height = 321; // 16 * 28 + 1
            this.width = document.documentElement.clientWidth - 400;
            this.height = parseInt(this.width / 3);
            this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
            this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
            this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        } else {
            this.width = document.documentElement.clientWidth - 50;
            this.height = parseInt(this.width / 2);
            this.canvas.addEventListener("touchstart", this.onTouchStart.bind(this));
            this.canvas.addEventListener("touchmove", this.onTouchMove.bind(this));
            this.canvas.addEventListener("touchend", this.onTouchEnd.bind(this));
            console.log(123)
        }
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        this.ctx = this.canvas.getContext('2d');
        this.initialize();
    }
    initialize() {
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, this.width, this.height);
        // this.drawInput();
        $('#output td').text('').removeClass('success');
    }
    onMouseDown(e) {
        this.canvas.style.cursor = 'default';
        this.drawing = true;
        this.prev = this.getPosition(e.clientX, e.clientY);
    }
    onMouseUp() {
        this.drawing = false;
        // this.drawInput();
    }
    onMouseMove(e) {
        if (this.drawing) {
            var curr = this.getPosition(e.clientX, e.clientY);
            this.ctx.lineWidth = 5;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();
            this.ctx.moveTo(this.prev.x, this.prev.y);
            this.ctx.lineTo(curr.x, curr.y);
            this.ctx.stroke();
            this.ctx.closePath();
            this.prev = curr;
        }
    }
    onTouchStart(e) {
        this.canvas.style.cursor = 'default';
        this.drawing = true;
        this.prev = this.getPosition(e.changedTouches[0].clientX, e.changedTouches[0].clientY);
    }
    onTouchMove(e) {
        e.preventDefault();
        if (this.drawing) {
            var curr = this.getPosition(e.changedTouches[0].clientX, e.changedTouches[0].clientY);
            this.ctx.lineWidth = 3;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();
            this.ctx.moveTo(this.prev.x, this.prev.y);
            this.ctx.lineTo(curr.x, curr.y);
            this.ctx.stroke();
            this.ctx.closePath();
            this.prev = curr;
        }
    }
    onTouchEnd(e) {
        document.removeEventListener('touchmove', function (e) {
            e.preventDefault();
        }, false); //打开默认事件
    }
    getPosition(clientX, clientY) {
        var rect = this.canvas.getBoundingClientRect();
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }


    fileUPInput() {
        var dataurl = this.canvas.toDataURL("image/png");
        var arr = dataurl.split(','),
            mime = arr[0].match(/:(.*?);/)[1],
            bstr = atob(arr[1]),
            n = bstr.length,
            u8arr = new Uint8Array(n);
        while (n--) {
            u8arr[n] = bstr.charCodeAt(n);
        }
        var blob = new Blob([u8arr], {
            type: mime
        });

        var fd = new FormData();
        fd.append("image", blob, "image.png");

        var datas = fd;
        // alert(inputs)
        $.ajax({
            url: 'https://wangjiayi.fun/api/ocr/multiple_predict',
            // url: 'http://127.0.0.1:8000/multiple_predict',
            method: 'Post',
            type: 'POST',
            processData: false,
            contentType: false,
            // contentType: 'application/json',
            data: datas,
            success: (data) => {
                console.log(data);
                $('#result-multiple')[0].innerHTML = data.result[0];
                for (let i = 1; i < data.result.length; i++) {
                    if (data.result[i] != '') {
                        $('#result-text').append('<div style=\"margin-top:5px;\" id=\"result-multiple\">' + data.result[i] +
                            '</div>');
                    }
                }
                var base_str = "data:image/jpeg;base64," + data.img;
                console.log(base_str);
                $('#result-img')[0].setAttribute('src', base_str);
            }
        });
    }
}

$(() => {
    var main = new Main();
    layui.use(['upload', 'layer'], function () {
        var upload = layui.upload;
        var layer = layui.layer;
        var uploadInst = upload.render({
            elem: '#upload', //绑定元素
            url: 'https://wangjiayi.fun/api/ocr/upload',
            // url: 'http://127.0.0.1:8000/upload',
            before: function (obj) {
                layer.msg('正在上传识别', {
                    icon: 16,
                    shade: 0.01
                });
            },
            done: function (data) {
                console.log(data);
                $('#result-multiple')[0].innerHTML = data.result[0];
                for (let i = 1; i < data.result.length; i++) {
                    if (data.result[i] != '') {
                        $('#result-text').append(
                            '<div style=\"margin-top:5px;\" id=\"result-multiple\">' + data
                            .result[i] +
                            '</div>');
                    }
                }
                var base_str = "data:image/jpeg;base64," + data.img;
                // console.log(base_str);
                $('#result-img')[0].setAttribute('src', base_str);
                layer.closeAll();
            },
            error: function () {
                //请求异常回调
            }
        });

    });

    $('#clear').click(() => {
        console.log(999)
        main.initialize();
        $('#result-text')[0].innerHTML = '<p class=\"result-title\">文字识别结果:</p> <div id=\"result-multiple\"></div>';
        $('#result-img')[0].setAttribute('src', "");
    });
    $('#text').click(() => {
        main.fileUPInput()
    });


});