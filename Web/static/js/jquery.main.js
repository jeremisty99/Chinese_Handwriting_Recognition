/* global $ */
class Main {
    constructor() {
        this.canvas = document.getElementById('main');
        this.input = document.getElementById('input');
        if (document.documentElement.clientWidth > 768) {
            this.width = 449; // 16 * 28 + 1
            this.height = 449; // 16 * 28 + 1
            this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
            this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
            this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        } else {
            this.width = document.documentElement.clientWidth - 50;
            this.height = document.documentElement.clientWidth - 50;
            this.canvas.addEventListener("touchstart", this.onTouchStart.bind(this));
            this.canvas.addEventListener("touchmove", this.onTouchMove.bind(this));
            this.canvas.addEventListener("touchend", this.onTouchEnd.bind(this));
        }
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        this.ctx = this.canvas.getContext('2d');
        this.initialize();
    }
    initialize() {
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, this.width, this.height);
        this.drawInput();
        $('#output td').text('').removeClass('success');
    }
    onMouseDown(e) {
        this.canvas.style.cursor = 'default';
        this.drawing = true;
        this.prev = this.getPosition(e.clientX, e.clientY);
    }
    onMouseUp() {
        this.drawing = false;
        this.drawInput();
    }
    onMouseMove(e) {
        if (this.drawing) {
            var curr = this.getPosition(e.clientX, e.clientY);
            this.ctx.lineWidth = 10;
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
        // console.log(this.prev);
    }
    onTouchMove(e) {
        e.preventDefault();
        if (this.drawing) {
            // console.log("move");
            var curr = this.getPosition(e.changedTouches[0].clientX, e.changedTouches[0].clientY);
            this.ctx.lineWidth = 6;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();
            this.ctx.moveTo(this.prev.x, this.prev.y);
            this.ctx.lineTo(curr.x, curr.y);
            this.ctx.stroke();
            this.ctx.closePath();
            this.prev = curr;
        }
        this.drawInput();
    }
    onTouchEnd(e) {
        this.drawInput();
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
    drawInput() {
        var ctx = this.input.getContext('2d');
        var img = new Image();
        img.onload = () => {
            var small = document.createElement('canvas').getContext('2d');
            small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 64, 64);
            var data = small.getImageData(0, 0, 64, 64).data;
            // console.log(data);
            for (var i = 0; i < 64; i++) {
                for (var j = 0; j < 64; j++) {
                    var n = 4 * (i * 64 + j);
                    ctx.fillStyle = 'rgb(' + [data[n + 0], data[n + 1], data[n + 2]].join(',') + ')';
                    ctx.fillRect(j * 1, i * 1, 1, 1);
                }
            }
        };
        img.src = this.canvas.toDataURL();
    }


    fileUPInput() {
        var dataurl = this.input.toDataURL("image/png");
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
            url: 'http://127.0.0.1:8000/single_predict',
            method: 'Post',
            type: 'POST',
            processData: false,
            contentType: false,
            // contentType: 'application/json',
            data: datas,
            success: (data) => {
                console.log(data);
                $('#result')[0].innerHTML = data.result;
                for (let i = 0; i < 10; i++) {
                    $('#output tr').eq(i + 1).find('td').eq(0).text(data.list[i][0]);
                    $('#output tr').eq(i + 1).find('td').eq(1).text(data.list[i][1]);
                }
            }
        });
    }
}

$(() => {
    var main = new Main();
    $('#clear').click(() => {
        main.initialize();
        $("#result")[0].innerHTML = '';
    });
    $('#text').click(() => {
        main.fileUPInput()
    });


});