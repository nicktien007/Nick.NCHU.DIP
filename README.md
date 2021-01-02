# 數位影像處理

## 透視變形校正 Perspective Distortion Correction

輸入一張圖，輸入x1,x2,x3,x4,y1,y2,y3,y4 八個點位做透視變形校正

參數

position1: `input_image_path`
 
position2: `x1,x2,x3,x4,y1,y2,y3,y4`


調用範例：
```
python perspective.py ./test/Perspective_4.png 258,247,650,633,271,555,261,534
```

執行結果

![src](./test/Perspective_4.png)

![src](./output/Perspective_process_screenshot_03.01.2021.png)