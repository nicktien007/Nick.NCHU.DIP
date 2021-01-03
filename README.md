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

## 非銳化濾鏡 Unsharp Masking

參數

position1: `input_image_path`
 
position2: `threshold 閥值`


調用範例：
```
python unsharp.py ./test/Lena_2.png 128
```

執行結果

![src](./test/Lena_2.png)

![src](./output/Snipaste_20210103150104.png)

## 邊界抽取 Boundary Extraction

參數

position1: `input_image_path`
 
position2: `element_size`


調用範例：
```
python boundary.py ./test/staff.png 3
```

執行結果
![result](./output/Snipaste_20210103152042.png)

## 區域填充 Region Filling

參數

position1: `input_image_path`
 
position2: `element_size`


調用範例：
```
python regionfilling.py ./test/RegionFilling.png 3
```

執行結果
![r1](./output/Snipaste_20210103153007.png)
![r2](./output/Snipaste_20210103153124.png)
![r3](./output/Snipaste_20210103153218.png)