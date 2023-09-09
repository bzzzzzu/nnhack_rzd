Минимальные системные требования - 24 GB VRAM / 32 GB RAM

Установка через Docker (Windows):

docker build -t fttftf_docker:gradio_gpu .
docker run --gpus all -it -p 7860:7860 fttftf_docker:gradio_gpu

Запуск через share link - "Running on public URL: https://1234567890abcd.gradio.live" или похожее, выведется в консоли при запуске докера.

В случае возникновения проблем может помочь перезагрузка веб-страницы градио в браузере, упавшая функция инференса не валит за собой все приложение. В противном случае перезапуск докера может быть очень длительный из-за загрузки всех моделей с нуля.

Локальная установка в Windows:

Все тоже самое, что и в докере, но установка торча и auto-gptq из закоменнченных строк requirements:
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# git clone https://github.com/PanQiWei/AutoGPTQ
# pip3 install AutoGPTQ/.
# ^ works on windows, another kind of dark magic required on linux, see dockerfile
# Successfully installed auto-gptq-0.5.0.dev0+cu117 gekko-1.0.6

Не рекомендуется, потому что слишком много подводных камней, которые я не могу отловить.

Установка в Linux/Docker (Linux):

Не проверялась.