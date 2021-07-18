# Deeplabv3plus_Reverse

## 모델 구조

![Screenshot from 2021-07-01 23-48-33](https://user-images.githubusercontent.com/76798025/124144434-ed077200-dac6-11eb-97e4-b0c1aef2437d.png)

![Screenshot from 2021-06-21 14-38-55](https://user-images.githubusercontent.com/76798025/122712222-88fed580-d29e-11eb-9313-b602d8708968.png)


## Download Dataset
[위성영상 객체 판독](https://aihub.or.kr/aidata/7982)

## 데이터 구성
```txt
/data
	/SIA
		/multi
			/gtFine
				/train
				/val
			/leftImg8bit
				/train
				/val
```

gtFine : label<br>
leftImg8bit : image


## Requirements
```bash
pip install -r requirements.txt
```

## Visdom

```bash
visdom -port 28333
```
